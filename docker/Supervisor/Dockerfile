FROM modynbase:latest

RUN chmod a+x /src/modyn/supervisor/modyn-supervisor
RUN mkdir -p /tmp/evaluation_results
RUN chown appuser /tmp/evaluation_results
RUN chmod -R 777 /tmp/evaluation_results
RUN mamba run -n modyn pip install peft
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn -a "" ./modyn/supervisor/modyn-supervisor ./modyn/config/examples/modyn_config.yaml
