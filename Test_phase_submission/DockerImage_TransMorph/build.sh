#!/usr/bin/env bash
echo "Building docker image..."
docker build -f Dockerfile -t reg_model .
