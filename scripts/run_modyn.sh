DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR=$(realpath ${DIR}/../)

/bin/bash "${DIR}/initial_setup.sh"
pushd $PARENT_DIR

docker compose down
docker build -t modyndependencies -f docker/Dependencies/Dockerfile .
docker build -t modynbase -f docker/Base/Dockerfile .
# APEX docker build -t modynapex -f docker/Apex/Dockerfile .
docker compose up -d --build supervisor

popd

echo "Modyn containers are running. Run 'docker compose down' to exit them. You can use 'tmuxp load tmuxp.yaml' to enter the containers easily using tmux."
