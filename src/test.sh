#!/bin/bash

python3 transcriber.py \
    -m $(pwd)/model-files/ \
    --websocket localhost:8082
