#!/bin/bash
exec python datafeeder/datafeeder/data_feeder.py ./config/devel.yaml &
exec python dataloader/dataloader/offline_data_loader.py ./config/devel.yaml
