#!/bin/bash

exec python ./data_feeder/data_feeder.py ./config/experiment1.yaml &
exec python /mvp.py
