#!/bin/bash
exec python ./modyn/backend/odm/odm_server.py ./modyn/config/config.yaml &
exec python ./modyn/backend/metadata_processor/metadata_processor.py ./modyn/config/config.yaml &
exec python ./modyn/backend/selector/selector_server.py ./modyn/config/config.yaml &
