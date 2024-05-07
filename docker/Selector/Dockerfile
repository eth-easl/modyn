FROM modynbase:latest

COPY ./modyn/selector/internal/storage_backend/local /src/modyn/selector/internal/storage_backend/local
COPY ./modyn/selector/CMakeLists.txt /src/modyn/selector/CMakeLists.txt
COPY ./modyn/selector/internal/CMakeLists.txt /src/modyn/selector/internal/CMakeLists.txt
COPY ./modyn/selector/internal/storage_backend/CMakeLists.txt /src/modyn/selector/internal/storage_backend/CMakeLists.txt

RUN chmod a+x /src/modyn/selector/modyn-selector
RUN mkdir -p /tmp/trigger_samples
RUN chown appuser /tmp/trigger_samples
RUN chmod -R 777 /tmp/trigger_samples

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn --no-capture-output ./modyn/selector/modyn-selector ./modyn/config/examples/modyn_config.yaml
