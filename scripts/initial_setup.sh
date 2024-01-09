DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR=$(realpath ${DIR}/../)
CONFIGURED_FILE="${PARENT_DIR}/.modyn_configured"

pushd $PARENT_DIR

# Check whether Modyn has already been configured for this system
if [[ -f "${CONFIGURED_FILE}" ]]; then
    exit 0
fi

CUDA_VERSION=12.1
# Make sure to use devel image!
CUDA_CONTAINER=nvidia/cuda:12.1.1-devel-ubuntu22.04
# container used in CI - cannot use cuda because it is too big.
CI_CONTAINER=python:3.10-slim

echo "This is the first time running Modyn, we're tweaking some nuts and bolts for your system."

cp environment.yml environment.yml.original

if [ "$(uname)" == "Darwin" ]; then
    IS_MAC=true
fi

# If ARM, adjust environment.yml
if [ "$IS_MAC" = true ] || [ "$(dpkg --print-architecture)" = "arm64" ] || [ "$(dpkg --print-architecture)" = "aarch64" ]; then
    echo "Detected ARM system, adjusting environment.yml"
    sed -i '' -e '/- nvidia/d' $PARENT_DIR/environment.yml # Delete Nvidia Channel
    sed -i '' -e 's/pytorch:://g' $PARENT_DIR/environment.yml # Do not use Pytorch Channel (broken)
fi

if [ "$IS_MAC" = true ]; then
    echo "Detected macOS, updating dev-requirements.txt to build grpcio from source"
    export GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation"
    sed -i '' -e 's/grpcio-tools/grpcio-tools --no-binary :all:/g' $PARENT_DIR/dev-requirements.txt 
    sed -i '' -e 's/grpcio # Linux/grpcio --no-binary :all:/g' $PARENT_DIR/dev-requirements.txt 
fi

# On CI, we change the base image from CUDA to Python and stop here
if [[ ! -z "$CI" ]]; then
    dockerContent=$(tail -n "+2" $PARENT_DIR/docker/Dependencies/Dockerfile)
    mv $PARENT_DIR/docker/Dependencies/Dockerfile $PARENT_DIR/docker/Dependencies/Dockerfile.original
    echo "FROM ${CI_CONTAINER}" > $PARENT_DIR/docker/Dependencies/Dockerfile
    echo "$dockerContent" >> $PARENT_DIR/docker/Dependencies/Dockerfile
    echo "Found CI server and set container to ${CI_CONTAINER}, exiting."
    touch "${CONFIGURED_FILE}"
    popd
    exit 0
fi

USING_CUDA=false

if [ "$IS_MAC" != true ]; then
    MAYBE_CUDA=false
    if [[ $(ls -l /usr/local | grep cuda) ]]; then
        MAYBE_CUDA=true
    fi

    if [ command -v nvidia-smi &> /dev/null ]; then
        MAYBE_CUDA=true
    fi

    if [ "$MAYBE_CUDA" = true ] ; then
        echo 'We suspect you are running on a GPU machine.'
    else
        echo 'We did not find a GPU on this system; this may be a false-negative, however.'
    fi

    use_cuda () {
        USING_CUDA=true
        # Fix environment.yml
        sed -i '/cpuonly/d' $PARENT_DIR/environment.yml # Delete cpuonly
        sed -i "/pytorch-cuda/c\  - pytorch::pytorch-cuda=$CUDA_VERSION" $PARENT_DIR/environment.yml # Enable pytorch-cuda
        sed -i "/nvidia::cuda-libraries-dev/c\  - nvidia::cuda-libraries-dev=$CUDA_VERSION.*" $PARENT_DIR/environment.yml
        sed -i "/nvidia::cuda-nvcc/c\  - nvidia::cuda-nvcc=$CUDA_VERSION.*" $PARENT_DIR/environment.yml
        sed -i "/nvidia::cuda-nvtx/c\  - nvidia::cuda-nvtx=$CUDA_VERSION.*" $PARENT_DIR/environment.yml
        sed -i "/nvidia::cuda-cupti/c\  - nvidia::cuda-cupti=$CUDA_VERSION.*" $PARENT_DIR/environment.yml

        # Fix $PARENT_DIR/docker-compose.yml
        startLine="$(grep -n "CUDASTART" $PARENT_DIR/docker-compose.yml | head -n 1 | cut -d: -f1)"
        endLine="$(grep -n "CUDAEND" $PARENT_DIR/docker-compose.yml | head -n 1 | cut -d: -f1)"
        fileLines="$(wc -l $PARENT_DIR/docker-compose.yml | awk '{ print $1 }')"
        
        fileBegin=$(tail -n "+1" $PARENT_DIR/docker-compose.yml | head -n $((${startLine}-1+1)))
        let tmp=$fileLines-$endLine+2
        fileEnd=$(tail -n "+${endLine}" $PARENT_DIR/docker-compose.yml | head -n $tmp)
        let trueStart=$startLine+1
        let tmp=$endLine-$trueStart
        fileCuda=$(tail -n "+${trueStart}" $PARENT_DIR/docker-compose.yml | head -n $tmp)

        cudaFixed=$(echo "$fileCuda" | sed "s/#//")

        mv $PARENT_DIR/docker-compose.yml $PARENT_DIR/docker-compose.yml.original
        echo "$fileBegin" > $PARENT_DIR/docker-compose.yml
        echo "$cudaFixed" >> $PARENT_DIR/docker-compose.yml
        echo "$fileEnd" >> $PARENT_DIR/docker-compose.yml

        # Fix Dockerfile
        dockerContent=$(tail -n "+2" $PARENT_DIR/docker/Dependencies/Dockerfile)
        mv $PARENT_DIR/docker/Dependencies/Dockerfile $PARENT_DIR/docker/Dependencies/Dockerfile.original
        echo "FROM ${CUDA_CONTAINER}" > $PARENT_DIR/docker/Dependencies/Dockerfile
        echo "$dockerContent" >> $PARENT_DIR/docker/Dependencies/Dockerfile
    }

    while true; do
        read -p "Do you want to use CUDA for Modyn? (y/n) " yn
        case $yn in
            [Yy]* ) use_cuda; break;;
            [Nn]* ) break;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    use_apex () {
        cp $PARENT_DIR/docker/Trainer_Server/Dockerfile $PARENT_DIR/docker/Trainer_Server/Dockerfile.original
        # Enable apex in Dockerfiles
        sed -i 's/# FROM/FROM/' $PARENT_DIR/docker/Evaluator/Dockerfile # Evaluator
        sed -i 's/# COPY/COPY/' $PARENT_DIR/docker/Evaluator/Dockerfile # Evaluator
        sed -i 's/# FROM/FROM/' $PARENT_DIR/docker/Trainer_Server/Dockerfile # Trainer Server
        sed -i 's/# COPY/COPY/' $PARENT_DIR/docker/Trainer_Server/Dockerfile # Trainer Server
        sed -i 's/# FROM/FROM/' $PARENT_DIR/docker/Model_Storage/Dockerfile # Model Storage
        sed -i 's/# COPY/COPY/' $PARENT_DIR/docker/Model_Storage/Dockerfile # Model Storage

        sed -i 's/# APEX//' $PARENT_DIR/scripts/run_integrationtests.sh
        sed -i 's/# APEX//' $PARENT_DIR/scripts/run_modyn.sh

        runtime=$(docker info | grep "Default Runtime")
        if [[ $runtime != *"nvidia"* ]]; then
            # Make nvidia runtime the default 
            echo "Apex required CUDA during container build. This is only possible by making NVIDIA the default docker runtime. Changing."
            sudo nvidia-ctk runtime configure
            pushd $(mktemp -d)
            (sudo cat /etc/docker/daemon.json 2>/dev/null || echo '{}') | \
                jq '. + {"default-runtime": "nvidia"}' | \
                tee tmp.json
            sudo mv tmp.json /etc/docker/daemon.json
            popd
            echo "Set default runtime, restarting docker"
            sudo systemctl restart docker
            echo "Docker restarted"
        else
            echo "NVIDIA runtime already is default"
        fi
    }

    if [ "$USING_CUDA" = true ] ; then
        while true; do
            read -p "Do you want to use Apex for Modyn (requires sudo)? (Takes a long time for initial Docker build, but required e.g. for DLRM model) (y/n) " yn
            case $yn in
                [Yy]* ) use_apex; break;;
                [Nn]* ) break;;
                * ) echo "Please answer yes or no.";;
            esac
        done
    fi
fi

echo "Successfully configured Modyn. Make sure to add mounts for datasets and fast temporary storage for the selector and enable the shm_size options, if required."
touch "${CONFIGURED_FILE}"

popd