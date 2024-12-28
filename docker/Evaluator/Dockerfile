# FROM modynapex:latest as apex-image

FROM modynbase:latest AS evaluatorimage

# COPY --from=apex-image /opt/mamba/envs/modyn /opt/mamba/envs/modyn

RUN chmod a+x /src/modyn/evaluator/modyn-evaluator

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD mamba run -n modyn -a "" ./modyn/evaluator/modyn-evaluator ./modyn/config/examples/modyn_config.yaml
