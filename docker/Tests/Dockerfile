FROM modynbase:latest

USER root
RUN mkdir -p /app/storage
RUN chown appuser /app/storage
RUN mkdir -p /app/model_storage
RUN chown appuser /app/model_storage

USER appuser
RUN chmod a+x /src/integrationtests/run.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn -a "" /src/integrationtests/run.sh
