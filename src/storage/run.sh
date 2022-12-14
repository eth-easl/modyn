#!/bin/bash
exec python ./storage/storage_server.py ./config/config.yaml &
exec python ./storage/data_sourcer.py ./config/config.yaml