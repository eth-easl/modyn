FROM modynbase

RUN chmod a+x /src/modyn/trainer_server/modyn-trainer-server

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/trainer_server/modyn-trainer-server ./modyn/config/examples/modyn_config.yaml