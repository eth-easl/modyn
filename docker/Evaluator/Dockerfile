FROM modynbase:latest

RUN chmod a+x /src/modyn/evaluator/modyn-evaluator

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn --no-capture-output ./modyn/evaluator/modyn-evaluator ./modyn/config/examples/modyn_config.yaml