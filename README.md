# Modyn: A Research Platform For ML Model Training On Dynamic Datasets

TODO(MaxiBoether): Improve the README.md

## Contributing

How to [contribute](CONTRIBUTING.md).

## Development 

### Requirements:
- [Docker](https://docs.docker.com/get-docker/) for the infrastructure, such as databases and storage
- For development, you might want to install the local dev requirements using `pip install -r dev-requirements.txt` when in the project root
- Furthermore, to install modyn as a development module, i.e., a module that can be imported but is synced to the current source code in the folder, run `pip install -e .` in the project root

### Running pytest and flake8 locally

Make sure that you installed modyn as a development module as stated above. Then, in the project root you can run `flake8 --statistics` and `pytest` to run unit tests and the flake8 linter.

### Running autopep8

autopep8 is an automatic linter for pep8 compliance that can fix many issues automatically. Make sure to commit/backup before running autopep8 inplace. Then, run
```
autopep8 modyn --recursive --in-place --pep8-passes 2000 --verbose
```
in the project root.

### Configuration:
- In `src/config/config.yaml` you will find a test configuration for the system, adapt as required

### Current architecture:

![Current architecture diagram](docs/images/Architecture.png)

### How to run:
1. Install [Docker](https://docs.docker.com/get-docker/) 
2. Run docker `docker compose -f "modyn/docker-compose.yml" up -d --build`
3. To read output [ssh](https://phase2.github.io/devtools/common-tasks/ssh-into-a-container/) into the docker container you are interested in
