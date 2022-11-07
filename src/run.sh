#!/bin/bash
exec python dynamicdatasets/feeder/feeder.py ./config/devel.yaml &&
exec python dynamicdatasets/metadata/metadata_server.py ./config/devel.yaml &&
exec python dynamicdatasets/offline/offline_preprocessor.py ./config/devel.yaml 
