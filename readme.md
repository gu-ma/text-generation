# Text Generation API with Hugginface Transformers

## Installation

### With pip

This repository is tested on Python , Flax 0., PyTorch 1. and TensorFlow 2.3+.

You need to install at least one of Flax, PyTorch or TensorFlow to use HF Transformers.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/), [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) and/or [Flax](https://github.com/google/flax#quick-install) and [Jax](https://github.com/google/jax#installation) installation pages regarding the specific installation command for your platform.

Then install as follow:

```bash
pip install -r requirements.txt
pip install transformers
```

### With conda

Since Transformers version v4.0.0, there is a conda channel: `huggingface`. 🤗 Transformers can be installed using conda as follows:

```shell script
conda install -c huggingface transformers
```

Follow the installation pages of Flax, PyTorch or TensorFlow to see how to install them with conda.

> **_NOTE:_**  On Windows, you may be prompted to activate Developer Mode in order to benefit from caching. If this is not an option for you, please let us know in [this issue](https://github.com/huggingface/huggingface_hub/issues/1062).

### Using Docker (Windows)

```bash
.\docker_build.bat
```

## Run API

### Using command line

```bash
python api.py
```

See [api.py](api.py) for a list of command line args

### Using Docker

```bash
.\docker_run.bat arg1 arg2
```

for example:

```bash
.\docker_run.bat -d --restart=unless-stopped
```

Or

```bash
.\docker_run.bat --rm
```

## API

`http://127.0.0.1:5004/generate_text?prompt=my prompt&length=50&temperature=1&num=1`  
Generate text
