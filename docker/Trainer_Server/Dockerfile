FROM modyndependencies:latest as apex-image
# We rely on modyndependencies so we only need to re-install apex when the dependencies change, not when the source code of Modyn changes
# Uncomment the following lines to install apex

# TODO(#104): Make this easily configurable here
# RUN mamba run -n modyn pip install packaging
# RUN git clone https://github.com/NVIDIA/apex ./apex
# RUN mamba run -v -n modyn pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

FROM modynbase:latest AS trainerserver-image

COPY --from=apex-image /opt/conda/envs/modyn /opt/conda/envs/modyn
RUN chmod a+x /src/modyn/trainer_server/modyn-trainer-server
RUN mkdir -p /tmp/offline_dataset
RUN chown appuser /tmp/offline_dataset

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn --no-capture-output ./modyn/trainer_server/modyn-trainer-server ./modyn/config/examples/modyn_config.yaml
