# Check whether Modyn has already been configured for this system
if [[ -f ".modyn_configured" ]]; then
    exit 0
fi

CUDA_VERSION=11.7
CUDA_CONTAINER=nvidia/cuda:11.7.1-devel-ubuntu22.04

echo "This is the first time running Modyn, we're tweaking some nuts and bolts for your system."
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cp environment.yml environment.yml.original

if [ "$(uname)" == "Darwin" ]; then
    IS_MAC=true
fi

# If ARM, adjust environment.yml
if [ "$IS_MAC" = true ] || [ "$(dpkg --print-architecture)" = "arm64" ] || [ "$(dpkg --print-architecture)" = "aarch64" ]; then
    echo "Detected ARM system, adjusting environment.yml"
    sed -i '' -e '/- nvidia/d' $SCRIPT_DIR/environment.yml # Delete Nvidia Channel
    sed -i '' -e 's/pytorch:://g' $SCRIPT_DIR/environment.yml # Do not use Pytorch Channel (broken)
fi

# On CI, we stop here
if [[ ! -z "$CI" ]]; then
    echo "Found CI server, exiting."
    touch ".modyn_configured"
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
        sed -i '/cpuonly/d' $SCRIPT_DIR/environment.yml # Delete cpuonly
        sed -i "/nvidia::cudatoolkit/c\  - nvidia::cudatoolkit=$CUDA_VERSION" $SCRIPT_DIR/environment.yml # Enable cudatoolkit
        sed -i "/pytorch-cuda/c\  - pytorch::pytorch-cuda=$CUDA_VERSION" $SCRIPT_DIR/environment.yml # Enable pytorch-cuda

        # Fix $SCRIPT_DIR/docker-compose.yml
        startLine="$(grep -n "CUDASTART" $SCRIPT_DIR/docker-compose.yml | head -n 1 | cut -d: -f1)"
        endLine="$(grep -n "CUDAEND" $SCRIPT_DIR/docker-compose.yml | head -n 1 | cut -d: -f1)"
        fileLines="$(wc -l $SCRIPT_DIR/docker-compose.yml | awk '{ print $1 }')"
        
        fileBegin=$(tail -n "+1" $SCRIPT_DIR/docker-compose.yml | head -n $((${startLine}-1+1)))
        let tmp=$fileLines-$endLine+1
        fileEnd=$(tail -n "+${endLine}" $SCRIPT_DIR/docker-compose.yml | head -n $tmp)
        let trueStart=$startLine+1
        let tmp=$endLine-$trueStart
        fileCuda=$(tail -n "+${trueStart}" $SCRIPT_DIR/docker-compose.yml | head -n $tmp)

        cudaFixed=$(echo "$fileCuda" | sed "s/#//")

        mv $SCRIPT_DIR/docker-compose.yml $SCRIPT_DIR/docker-compose.yml.original
        echo $fileBegin > $SCRIPT_DIR/docker-compose.yml
        echo $cudaFixed >> $SCRIPT_DIR/docker-compose.yml
        echo $fileEnd >> $SCRIPT_DIR/docker-compose.yml

        # Fix Dockerfile
        dockerContent=$(tail -n "+1" $SCRIPT_DIR/docker/Dependencies/Dockerfile)
        mv $SCRIPT_DIR/docker/Dependencies/Dockerfile $SCRIPT_DIR/docker/Dependencies/Dockerfile.original
        echo "FROM ${CUDA_CONTAINER}" > $SCRIPT_DIR/docker/Dependencies/Dockerfile
        echo "$dockerContent" >> $SCRIPT_DIR/docker/Dependencies/Dockerfile
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
        cp $SCRIPT_DIR/docker/Trainer_Server/Dockerfile $SCRIPT_DIR/docker/Trainer_Server/Dockerfile.original
        sed -i 's/# RUN/RUN/' $SCRIPT_DIR/docker/Trainer_Server/Dockerfile
    }

    if [ "$USING_CUDA" = true ] ; then
        while true; do
            read -p "Do you want to use Apex for Modyn? (Takes a long time for initial Docker build, but required e.g. for DLRM model) (y/n) " yn
            case $yn in
                [Yy]* ) use_apex; break;;
                [Nn]* ) break;;
                * ) echo "Please answer yes or no.";;
            esac
        done
    fi
fi

echo "Successfully configured Modyn."
touch ".modyn_configured"