FROM modynbase:latest

RUN chmod a+x /src/integrationtests/run.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn --no-capture-output /src/integrationtests/run.sh