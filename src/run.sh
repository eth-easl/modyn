#!/bin/bash
cd ..
exec python ./src/datafeeder/data_feeder.py ./config/experiment1.yaml &
exec python ./src/dataloader/data_loader.py ./config/experiment1.yaml
