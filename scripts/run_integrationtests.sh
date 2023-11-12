DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR=$(realpath ${DIR}/../)

/bin/bash "${DIR}/initial_setup.sh"
pushd $PARENT_DIR

docker compose down

BUILDTYPE=${1:-Release}
echo "Using build type ${BUILDTYPE} for integrationtests."
if [[ "$BUILDTYPE" == "Release" ]]; then
    DEPBUILDTYPE="Release"
else
    # Since Asan/Tsan are not necessarily targets of dependencies, we switch to debug mode in all other cases.
    DEPBUILDTYPE="Debug"
fi

echo "Inferred dependency buildtype ${DEPBUILDTYPE}."

# When on Github CI, we use the default postgres config to not go OOM
if [[ ! -z "$CI" ]]; then
    mv conf/metadata_postgresql.conf conf/metadata_postgresql.conf.bak
    mv conf/storage_postgresql.conf conf/storage_postgresql.conf.bak
    cp conf/default_postgresql.conf conf/metadata_postgresql.conf
    cp conf/default_postgresql.conf conf/storage_postgresql.conf
fi

docker build -t modyndependencies -f docker/Dependencies/Dockerfile --build-arg MODYN_BUILDTYPE=$BUILDTYPE --build-arg MODYN_DEP_BUILDTYPE=$DEPBUILDTYPE .
docker build -t modynbase -f docker/Base/Dockerfile .
docker compose up --build tests --abort-on-container-exit --exit-code-from tests

exitcode=$?

echo "LOGS START"
echo "METADATADB"
docker logs $(docker compose ps -q metadata-db)
echo "STORAGEDB"
docker logs $(docker compose ps -q storage-db)
echo "STORAGE"
docker logs $(docker compose ps -q storage)
echo "SELECTOR"
docker logs $(docker compose ps -q selector)
echo "TRAINERSERVER"
docker logs $(docker compose ps -q trainer_server)
echo "LOGS END"

# Cleanup
docker compose down
if [[ ! -z "$CI" ]]; then
    rm conf/storage_postgresql.conf conf/metadata_postgresql.conf
    mv conf/metadata_postgresql.conf.bak conf/metadata_postgresql.conf
    mv conf/storage_postgresql.conf.bak conf/storage_postgresql.conf
fi

popd

exit $exitcode
