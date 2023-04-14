docker compose down
docker build -t modynbase -f docker/Base/Dockerfile .
docker compose up -d --build supervisor

echo "Modyn containers are running. Run `docker compose down` to exit them. You can use `tmuxp load tmuxp.yaml` to enter the containers easily using tmux."