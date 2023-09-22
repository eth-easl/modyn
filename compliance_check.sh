echo "Running compilance check"

if ! command -v mamba &> /dev/null
then
    echo "Cannot find mamba."
    if [ -n "$ZSH_VERSION" ]; then
        source ~/.zshrc
        eval "$(mamba shell.zsh hook)"
    elif [ -n "$BASH_VERSION" ]; then
        source ~/.bashrc
        eval "$(mamba shell.bash hook)"
    else
        echo "Unknown shell, neither zsh nor bash, please ensure mamba is available before running the script."
        exit 1
    fi
    
    if ! command -v mamba &> /dev/null
    then
        echo "mamba still not available after trying to fix, please ensure mamba is available before running the script."
        exit 1
    fi
fi

echo "Running auto-formatters"

mamba run -n modyn isort . > /dev/null
mamba run -n modyn autopep8 modyn integrationtests --recursive --in-place --pep8-passes 2000 > /dev/null
mamba run -n modyn black modyn integrationtests --verbose --config black.toml > /dev/null

echo "Running linters"

if mamba run -n modyn flake8 ; then
    echo "No flake8 errors"
else
    echo "flake8 errors"
    exit 1
fi

if mamba run -n modyn isort . --check --diff ; then
    echo "No isort errors"
else
    echo "isort errors"
    exit 1
fi

if mamba run -n modyn black --check modyn --config black.toml ; then
    echo "No black errors"
else
    echo "black errors"
    exit 1
fi

if mamba run -n modyn pylint modyn ; then
    echo "No pylint errors"
else
    echo "pylint errors"
    exit 1
fi

echo "Running tests"

if mamba run -n modyn pytest ; then
    echo "No pytest errors"
else
    echo "pytest errors"
    exit 1
fi

if mamba run -n modyn mypy modyn ; then
    echo "No mypy errors"
else
    echo "mypy errors"
    exit 1
fi

echo "Successful compliance check"
