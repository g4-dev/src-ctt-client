#!/bin/bash

python3 transcriber.py \
    -m $(pwd)/model-files/ \
    --websocket ctt-server.loicroux.com:82
