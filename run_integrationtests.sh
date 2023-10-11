/bin/bash initial_setup.sh
docker compose down

# When on Github CI, we use the default postgres config to not go OOM
if [[ ! -z "$CI" ]]; then
    mv metadata_postgresql.conf metadata_postgresql.conf.bak
    mv storage_postgresql.conf storage_postgresql.conf.bak
    cp default_postgresql.conf metadata_postgresql.conf
    cp default_postgresql.conf storage_postgresql.conf
fi

docker build -t modyndependencies -f docker/Dependencies/Dockerfile .
docker build -t modynbase -f docker/Base/Dockerfile .
docker compose up --build tests --abort-on-container-exit --exit-code-from tests --attach storage
exitcode=$?

# Cleanup
docker compose down
if [[ ! -z "$CI" ]]; then
    rm storage_postgresql.conf metadata_postgresql.conf
    mv metadata_postgresql.conf.bak metadata_postgresql.conf
    mv storage_postgresql.conf.bak storage_postgresql.conf
fi

exit $exitcode
