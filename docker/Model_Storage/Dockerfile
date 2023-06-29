FROM modynbase:latest

RUN chmod a+x /src/modyn/model_storage/modyn-model-storage

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn --no-capture-output ./modyn/model_storage/modyn-model-storage ./modyn/config/examples/modyn_config.yaml