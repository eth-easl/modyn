#!/bin/zsh
echo "Running compilance check"

echo "Running auto-formatters"

source ~/.zshrc

eval "$(conda shell.zsh hook)"
conda activate modyn > /dev/null

isort . > /dev/null
autopep8 modyn --recursive --in-place --pep8-passes 2000 > /dev/null
black modyn --verbose --config black.toml > /dev/null

echo "Running linters"

if flake8 ; then
    echo "No flake8 errors"
else
    echo "flake8 errors"
    exit 1
fi

if isort . --check --diff ; then
    echo "No isort errors"
else
    echo "isort errors"
    exit 1
fi

if black --check modyn --config black.toml ; then
    echo "No black errors"
else
    echo "black errors"
    exit 1
fi

if pylint modyn ; then
    echo "No pylint errors"
else
    echo "pylint errors"
    exit 1
fi

echo "Running tests"

if pytest ; then
    echo "No pytest errors"
else
    echo "pytest errors"
    exit 1
fi

if mypy modyn ; then
    echo "No mypy errors"
else
    echo "mypy errors"
    exit 1
fi

echo "Successfull compilance check"
