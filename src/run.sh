#!/bin/bash
exec python datafeeder/datafeeder/data_feeder.py ./config/devel.yaml &
exec python dataloader/dataloader/data_loader.py ./config/devel.yaml
