FROM modynbase:latest

RUN chmod a+x /src/modyn/backend/selector/modyn-selector

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/backend/selector/modyn-selector ./modyn/config/examples/modyn_config.yaml