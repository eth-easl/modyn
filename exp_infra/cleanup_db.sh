# !/bin/bash

docker compose up -d --build postgres-db-init
rm -r /scratch/jinzhu/modyn/postgres-data/*-test/*
ls /scratch/jinzhu/modyn/postgres-data/*-test/*
~/run/docker_prune.sh
~/run/docker_prune.sh