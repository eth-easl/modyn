#!/bin/bash
exec python ./backend/newqueue/newqueue_server.py ./config/config.yaml &
exec python ./backend/odm/odm_server.py ./config/config.yaml &
exec python ./backend/ptmp/ptmp_server.py ./config/config.yaml &
exec python ./backend/selector/selector_server.py ./config/config.yaml &
exec python ./backend/newqueue/newqueue_poller.py ./config/config.yaml
