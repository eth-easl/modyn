echo "Running compliance check"

PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

pushd $PARENT_DIR

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


echo "Running auto-formatters"

$MAMBA_CMD run -n modyn isort . > /dev/null
$MAMBA_CMD run -n modyn autopep8 modyn integrationtests --recursive --in-place --pep8-passes 2000 > /dev/null
$MAMBA_CMD run -n modyn black modyn integrationtests --verbose --config black.toml > /dev/null

echo "Running linters"

if $MAMBA_CMD run -n modyn flake8 modyn ; then
    echo "No flake8 errors"
else
    echo "flake8 errors"
    exit 1
fi

if $MAMBA_CMD run -n modyn isort . --check --diff ; then
    echo "No isort errors"
else
    echo "isort errors"
    exit 1
fi

if $MAMBA_CMD run -n modyn black --check modyn --config black.toml ; then
    echo "No black errors"
else
    echo "black errors"
    exit 1
fi

if $MAMBA_CMD run -n modyn ruff check experiments benchmark/arxiv_kaggle benchmark/huffpost_kaggle ; then
    echo "No ruff check errors"
else
    echo "ruf check errors"
    exit 1
fi

if $MAMBA_CMD run -n modyn ruff format experiments benchmark/arxiv_kaggle benchmark/huffpost_kaggle ; then
    echo "No ruff format errors"
else
    echo "ruff format errors"
    exit 1
fi

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
