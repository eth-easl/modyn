#!/bin/bash

experiment_name=$1
echo "$experiment_name"

# clean directories and make dbs writeable
rm -f /scratch/fdeaglio/trigger_samples/*
rm -f /scratch/fdeaglio/offline_dataset/*
mkdir "/scratch/fdeaglio/offline_dataset/"
chmod 777 /scratch/fdeaglio/offline_dataset/
bash ./run_modyn.sh
docker exec -it "$(docker compose ps -q storage-db)" chmod 777 /var/lib/postgresql/data
docker exec -it "$(docker compose ps -q metadata-db)" chmod 777 /var/lib/postgresql/data
docker exec -it "$(docker compose ps -q trainer_server)" bash -c "rm  ./TIME_LOGS/*"
docker exec -it "$(docker compose ps -q selector)" bash -c "rm  ./TIME_LOGS/*"

# run the pipeline
docker exec "$(docker compose ps -q supervisor)" bash -c " conda run -n modyn modyn-supervisor benchmark/pipeline_queue/$experiment_name.yaml modyn/config/examples/modyn_config.yaml tmpev --start-replay-at 0"

# download the models
rm -rf ../new_models/*
docker exec -it "$(docker compose ps -q storage)" bash -c "rm -f /src/core.*"
docker cp "$(docker compose ps -q model_storage)":/src/model_storage ../new_models/

# download the timings trainer server
rm -rf ../new_logs/*
docker exec -it "$(docker compose ps -q trainer_server)" bash -c "chmod 777 ./TIME_LOGS/*"
docker cp "$(docker compose ps -q trainer_server)":/src/TIME_LOGS ../new_logs/

# download the timings selector
rm -rf ../new_logs_selector/*
docker exec -it "$(docker compose ps -q selector)" bash -c "chmod 777 ./TIME_LOGS/*"
docker cp "$(docker compose ps -q selector)":/src/TIME_LOGS ../new_logs_selector/

# download the index mappings
rm -rf ../new_index_mappings/*
docker exec -it "$(docker compose ps -q selector)" bash -c "chmod 777 index_trigger_mapping.json"
docker cp "$(docker compose ps -q selector)":/src/index_trigger_mapping.json ../new_index_mappings/

docker compose down

# run the script to extract the data
python3 extract_data.py

# rename the processed file and backup on scratch
mv ../test_out/model_storage.pkl /scratch/fdeaglio/test_out_backup/"$experiment_name".pkl
mv ../new_models/model_storage /scratch/fdeaglio/new_models_backup/"$experiment_name"
mv ../new_logs/TIME_LOGS /scratch/fdeaglio/new_logs_backup/"$experiment_name"
mv ../new_logs_selector/TIME_LOGS /scratch/fdeaglio/new_logs_selector_backup/"$experiment_name"
mv ../new_index_mappings/index_trigger_mapping.json /scratch/fdeaglio/new_index_mappings_backup/"$experiment_name".json
mv /scratch/fdeaglio/offline_dataset/ /scratch/fdeaglio/offline_dataset_backup/"$experiment_name"