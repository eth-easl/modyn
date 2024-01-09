FROM modynbase:latest

RUN chmod a+x /src/modyn/model_storage/modyn-model-storage
RUN mkdir -p /tmp/models
RUN chown appuser /tmp/models
RUN chmod -R 777 /tmp/models

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn --no-capture-output ./modyn/model_storage/modyn-model-storage ./modyn/config/examples/modyn_config.yaml