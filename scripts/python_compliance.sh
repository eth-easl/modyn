echo "Running compilance check"

PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

pushd $PARENT_DIR

if ! command -v micromamba &> /dev/null
then
    echo "Cannot find micromamba."
    if [ -n "$ZSH_VERSION" ]; then
        source ~/.zshrc
        eval "$(micromamba shell.zsh hook)"
    elif [ -n "$BASH_VERSION" ]; then
        source ~/.bashrc
        eval "$(micromamba shell.bash hook)"
    else
        echo "Unknown shell, neither zsh nor bash, please ensure micromamba is available before running the script."
        exit 1
    fi
    
    if ! command -v micromamba &> /dev/null
    then
        echo "micromamba still not available after trying to fix, please ensure micromamba is available before running the script."
        exit 1
    fi
fi

echo "Running auto-formatters"

micromamba run -n modyn isort . > /dev/null
micromamba run -n modyn autopep8 modyn integrationtests --recursive --in-place --pep8-passes 2000 > /dev/null
micromamba run -n modyn black modyn integrationtests --verbose --config black.toml > /dev/null

echo "Running linters"

if micromamba run -n modyn flake8 modyn ; then
    echo "No flake8 errors"
else
    echo "flake8 errors"
    exit 1
fi

if micromamba run -n modyn isort . --check --diff ; then
    echo "No isort errors"
else
    echo "isort errors"
    exit 1
fi

if micromamba run -n modyn black --check modyn --config black.toml ; then
    echo "No black errors"
else
    echo "black errors"
    exit 1
fi

if micromamba run -n modyn pylint modyn ; then
    echo "No pylint errors"
else
    echo "pylint errors"
    exit 1
fi

echo "Running tests"

if micromamba run -n modyn pytest ; then
    echo "No pytest errors"
else
    echo "pytest errors"
    exit 1
fi

if micromamba run -n modyn mypy modyn ; then
    echo "No mypy errors"
else
    echo "mypy errors"
    exit 1
fi

echo "Successful compliance check"

popd
