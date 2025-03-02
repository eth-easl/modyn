# FROM modynapex:latest as apex-image

FROM modynbase:latest AS trainerserver-image

# COPY --from=apex-image /opt/mamba/envs/modyn /opt/mamba/envs/modyn
RUN chmod a+x /src/modyn/trainer_server/modyn-trainer-server
RUN mkdir -p /tmp/offline_dataset
RUN chown appuser /tmp/offline_dataset
RUN mamba run -n modyn pip install peft
RUN mamba run -n modyn pip install sentencepiece
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn -a "" ./modyn/trainer_server/modyn-trainer-server ./modyn/config/examples/modyn_config.yaml
