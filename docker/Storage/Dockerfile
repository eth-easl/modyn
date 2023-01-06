FROM modynbase

RUN chmod a+x /src/modyn/storage/run.sh

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn /src/modyn/storage/run.sh # TODO(MaxiBoether): replace with storage entrypoint as soon as storage PR is merged