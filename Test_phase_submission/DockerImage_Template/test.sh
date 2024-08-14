#!/usr/bin/env bash
#bash ./build.sh
docker load --input reg_model.tar.gz

docker run --rm  \
        --ipc=host \
        --memory 16g \
        --mount type=bind,source=[PATH for .json dataset file],target=/LUMIR_dataset.json \
        --mount type=bind,source=[Directory of input images],target=/input \
        --mount type=bind,source=[Directory of output predictions],target=/output \
        reg_model

