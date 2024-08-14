#!/usr/bin/env bash

bash ./build.sh

docker save reg_model | gzip -c > reg_model.tar.gz
