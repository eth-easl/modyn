DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR=$(realpath ${DIR}/../)
STORAGE_POSTGRES_DATA="${PARENT_DIR}/storage-postgres-data"
METADATA_POSTGRES_DATA="${PARENT_DIR}/metadata-postgres-data"

docker run -it --rm  -v "$STORAGE_POSTGRES_DATA":/tmp/hahaha postgres:15.2-alpine chmod -R 777 /tmp/hahaha
docker run -it --rm  -v "$METADATA_POSTGRES_DATA":/tmp/hahaha postgres:15.2-alpine chmod -R 777 /tmp/hahaha
rm -rf "$STORAGE_POSTGRES_DATA"
rm -rf "$METADATA_POSTGRES_DATA"
docker volume rm $(docker volume ls -q)
