FROM modyndependencies:latest as apex-image
# We rely on modyndependencies so we only need to re-install apex when the dependencies change, not when the source code of Modyn changes
# Uncomment the following lines to install apex

# TODO(#104): Make this easily configurable here
# RUN mamba run -n modyn pip install packaging ninja
# RUN git clone https://github.com/NVIDIA/apex ./apex
# RUN mamba run -v -n modyn pip install -v --no-build-isolation --no-cache-dir --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex


FROM modynbase:latest AS modelstorageimage 

# TODO(MaxiBoether): Refactor apex to have one dockerfile and use that when activated in model stoage, trainer server, and evaluator

COPY --from=apex-image /opt/mamba/envs/modyn /opt/mamba/envs/modyn
RUN chmod a+x /src/modyn/model_storage/modyn-model-storage
RUN mkdir -p /tmp/models
RUN chown appuser /tmp/models
RUN chmod -R 777 /tmp/models

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn --no-capture-output ./modyn/model_storage/modyn-model-storage ./modyn/config/examples/modyn_config.yaml