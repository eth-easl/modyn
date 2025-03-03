echo "Running compliance check"

PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

pushd $PARENT_DIR
source ~/.bashrc

if command -v mamba &> /dev/null
then
    MAMBA_CMD="mamba"
elif command -v micromamba &> /dev/null
then
    MAMBA_CMD="micromamba"
else
    echo "Neither mamba nor micromamba could be found. Please ensure one of them is available when running the script."
    echo "Checkout README.md for setup instruction."
    exit 1
fi

echo "Running additional linters"

if $MAMBA_CMD run -n modyn pylint modyn ; then
    echo "No pylint errors"
else
    echo "pylint errors"
    exit 1
fi

echo "Running tests"

if $MAMBA_CMD run -n modyn pytest ; then
    echo "No pytest errors"
else
    echo "pytest errors"
    exit 1
fi

if $MAMBA_CMD run -n modyn mypy modyn ; then
    echo "No mypy errors"
else
    echo "mypy errors"
    exit 1
fi

echo "Successful compliance check"

popd
