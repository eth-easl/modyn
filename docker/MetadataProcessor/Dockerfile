FROM modynbase

RUN chmod a+x /src/modyn/metadata_processor/modyn-metadata-processor

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/metadata_processor/modyn-metadata-processor ./modyn/config/examples/modyn_config.yaml