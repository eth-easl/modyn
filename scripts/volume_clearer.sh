
directory=/scratch/xianma/sigmod/modyntmp
# map the local path to /tmp/hahaha in the docker container and change permission to 777
docker run -it --rm  -v $directory:/tmp/hahaha postgres:15.2-alpine chmod -R 777 /tmp/hahaha

rm -rf $directory/*
rm .modyn_configured
docker volume rm $(docker volume ls -q)

mkdir -m 777 $directory/local_storage
mkdir -m 777 $directory/logs
mkdir -m 777 $directory/metadatadb5
mkdir -m 777 $directory/modelstorage
mkdir -m 777 $directory/offline_dataset
mkdir -m 777 $directory/storagedb2
mkdir -m 777 $directory/trigger_samples
