sh initial_setup.sh

docker compose down
docker build -t modyndependencies -f docker/Dependencies/Dockerfile .
docker build -t modynbase -f docker/Base/Dockerfile .
docker compose up -d --build supervisor

echo "Modyn containers are running. Run 'docker compose down' to exit them. You can use 'tmuxp load tmuxp.yaml' to enter the containers easily using tmux."
