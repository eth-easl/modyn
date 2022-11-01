#!/bin/bash
exec python data_feeder/datafeeder/data_feeder.py ./config/devel.yaml &
exec python offline_data_preprocessor/offlinedatapreprocessor/offline_data_preprocessor.py ./config/devel.yaml
