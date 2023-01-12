FROM modynbase

RUN chmod a+x /src/modyn/storage/modyn-storage

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml