#!/usr/bin/env bash
#bash ./build.sh
docker load --input reg_model.tar.gz

docker run --rm  \
        --ipc=host \
        --memory 16g \
        --mount type=bind,source=/scratch/jchen/python_projects/LUMIR_DockerContainer_Example/LUMIR_dataset.json,target=/LUMIR_dataset.json \
        --mount type=bind,source=/scratch/jchen/DATA/LUMIR,target=/input \
        --mount type=bind,source=/scratch/jchen/python_projects/LUMIR_output,target=/output \
        reg_model

