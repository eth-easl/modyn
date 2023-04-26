docker compose down
docker build -t modynbase -f docker/Base/Dockerfile .
docker compose up --build tests --abort-on-container-exit --exit-code-from tests
exitcode=$?
docker compose down
exit $exitcode