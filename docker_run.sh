docker run --network host -it --rm -v $(pwd):/scratch --user $(id -u):$(id -g) transformer-text-gen:latest bash -c \
"(cd /scratch/ && python api.py --path=models/salvatore)"
