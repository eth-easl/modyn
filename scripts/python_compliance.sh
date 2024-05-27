echo "Running compliance check"

PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

pushd $PARENT_DIR

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
