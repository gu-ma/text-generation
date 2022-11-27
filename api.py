# Based on: https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation
# Only work with GPT2 models

import re
import time
import string
import argparse
import logging

import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import spacy

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def clean_text(txt):
    # Remove line breaks
    txt = re.sub(r"(?<=[a-z])\r?\n", " ", txt)
    # Remove special char
    txt = re.sub(r"[^a-zA-Z0-9 ,.:;-]", "", txt)
    # Remove additional space
    txt = re.sub(" +", " ", txt)
    # Add caps
    txt = string.capwords(txt, sep=".")
    txt = string.capwords(txt, sep=". ")
    return txt


class GenerateText(Resource):
    def get(self):

        prompt_text = request.args.get("prompt")
        temperature = request.args.get("temp") if request.args.get("temp") else 1.0
        num_sequences = int(request.args.get("num")) if request.args.get("num") else 5
        length = int(request.args.get("length")) if request.args.get("length") else 20

        length = adjust_length_to_model(
            length, max_sequence_length=model.config.max_position_embeddings
        )

        if prompt_text:
            t1 = time.time()
            encoded_prompt = tokenizer.encode(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )
            encoded_prompt = encoded_prompt.to(args.device)

            input_ids = None if encoded_prompt.size()[-1] == 0 else encoded_prompt

            t2 = time.time() - t1
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=length + len(encoded_prompt[0]),
                temperature=temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=num_sequences,
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            gen_sequences = []
            gen_sentences = []

            t3 = time.time() - t1 - t2
            for gen_sequence_idx, gen_sequence in enumerate(output_sequences):
                print(f"Sequence -> {gen_sequence_idx + 1}")
                gen_sequence = gen_sequence.tolist()

                # Decode text
                text = tokenizer.decode(gen_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(args.stop_token) if args.stop_token else None]

                # Clean up
                text = clean_text(text)
                total_sequence = text

                # Detect sentences
                total_sentence = nlp(total_sequence)
                total_sentence = [str(sentence) for sentence in total_sentence.sents]
                for i, sentence in enumerate(total_sentence):
                    print(f"{i} - {sentence}")

                gen_sequences.append(total_sequence)
                gen_sentences.append(total_sentence)

            print(t2, t3, time.time() - t1 - t3, time.time() - t1)

            results = jsonify(
                prompt=prompt_text,
                sequences=gen_sequences,
                sentences=gen_sentences,
                duration=time.time() - t1
            )

        else:
            results = jsonify("Missing arguments")

        return results


api.add_resource(GenerateText, "/generate_text")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )

    parser.add_argument(
        "--stop_token",
        type=str,
        default=None,
        help="Token at which text generation is stopped",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="primarily useful for CTRL model; in that case, use 1.2",
    )

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Initialize the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.path)
    model = GPT2LMHeadModel.from_pretrained(args.path)
    model.to(args.device)

    if args.fp16:
        model.half()

    # NLP
    nlp = spacy.load("en_core_web_sm")

    logger.info(args)

    app.config["JSON_SORT_KEYS"] = False
    app.run(host="0.0.0.0", port=5004)
