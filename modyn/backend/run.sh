#!/bin/bash
exec python ./modyn/backend/odm/odm_server.py ./modyn/config/config.yaml &
exec python ./modyn/backend/ptmp/ptmp_server.py ./modyn/config/config.yaml &
exec python ./modyn/backend/selector/selector_server.py ./modyn/config/config.yaml &
