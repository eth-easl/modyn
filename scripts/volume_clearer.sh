
directory=/scratch/xianma/sigmod/modyntmp
# map the local path to /tmp/hahaha in the docker container and change permission to 777
docker run -it --rm  -v $directory:/tmp/hahaha postgres:15.2-alpine chmod -R 777 /tmp/hahaha

#rm -rf $directory/*
rm .modyn_configured
docker volume rm $(docker volume ls -q)

rm -rf $directory/local_storage
rm -rf $directory/logs
rm -rf $directory/metadatadb5
rm -rf $directory/modelstorage
rm -rf $directory/offline_dataset
#rm -rf $directory/storagedb2
rm -rf $directory/trigger_samples
