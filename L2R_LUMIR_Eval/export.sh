#!/usr/bin/env bash

bash ./build.sh

docker save l2rtest | gzip -c > L2RTest.tar.gz
