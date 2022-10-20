#!/bin/bash
exec python datafeeder/datafeeder/data_feeder.py ./config/experiment1.yaml &
exec python dataloader/dataloader/data_loader.py ./config/experiment1.yaml
