FROM modynbase:latest

# Verify CMake installation
RUN cmake --version

RUN mkdir -p ./modyn/storage/build \
    && cd ./modyn/storage/build \
    && cmake .. \
    && make -j8

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ./modyn/storage/build/modyn-storage ./modyn/config/examples/modyn_config.yaml