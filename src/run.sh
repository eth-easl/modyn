#!/bin/bash
exec python data_feeder/datafeeder/data_feeder.py ./config/devel.yaml &
exec python offline_data_loader/offlinedataloader/offline_data_loader.py ./config/devel.yaml
