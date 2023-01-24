#!/bin/bash
exec python ./modyn/backend/metadata_database/metadata_database_server.py ./modyn/config/config.yaml &
exec python ./modyn/backend/selector/selector_server.py ./modyn/config/config.yaml &
