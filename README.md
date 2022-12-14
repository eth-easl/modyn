# Dynamic Datasets DSL Project

## Dynamic Datasets Testing Infrastructure:

### Requirements:
- [Docker](https://docs.docker.com/get-docker/)

### Configuration:
- In `src/config/config.yaml` you will find a test configuration for the system, adapt as required

### Current architecture:

![Current architecture diagram](docs/images/Architecture.png)

### How to run:
1. Install [Docker](https://docs.docker.com/get-docker/) 
2. Run docker `docker compose -f "src/docker-compose.yml" up -d --build`
3. To read output [ssh](https://phase2.github.io/devtools/common-tasks/ssh-into-a-container/) into the docker container you are interested in
