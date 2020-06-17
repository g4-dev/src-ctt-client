#!/bin/bash

python3 transcriber.py \
    -m (pwd)/model-files/ \
    -w files/\
    --websocket localhost:8081/socket