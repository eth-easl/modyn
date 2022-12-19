#!/bin/bash
exec python ./modyn/storage/storage_server.py ./modyn/config/config.yaml &
exec python ./modyn/storage/data_sourcer.py ./modyn/config/config.yaml