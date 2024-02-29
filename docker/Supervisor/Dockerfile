FROM modynbase:latest

RUN chmod a+x /src/modyn/supervisor/modyn-supervisor

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn --no-capture-output ./modyn/supervisor/modyn-supervisor ./modyn/config/examples/modyn_config.yaml