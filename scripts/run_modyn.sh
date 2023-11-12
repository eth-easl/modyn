DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR=$(realpath ${DIR}/../)

/bin/bash "${DIR}/initial_setup.sh"
pushd $PARENT_DIR

docker compose down

BUILDTYPE=${1:-Release}
echo "Running Modyn with buildtype ${BUILDTYPE}."

if [[ "$BUILDTYPE" == "Release" ]]; then
    DEPBUILDTYPE="Release"
else
    # Since Asan/Tsan are not necessarily targets of dependencies, we switch to debug mode in all other cases.
    DEPBUILDTYPE="Debug"
fi

echo "Inferred dependency buildtype ${DEPBUILDTYPE}."

docker build -t modyndependencies -f docker/Dependencies/Dockerfile --build-arg MODYN_BUILDTYPE=$BUILDTYPE --build-arg MODYN_DEP_BUILDTYPE=$DEPBUILDTYPE  .
docker build -t modynbase -f docker/Base/Dockerfile .
docker compose up -d --build supervisor

popd

echo "Modyn containers are running. Run 'docker compose down' to exit them. You can use 'tmuxp load tmuxp.yaml' to enter the containers easily using tmux."
