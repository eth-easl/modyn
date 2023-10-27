FROM modynbase:latest

RUN mkdir -p ./modyn/storage/build \
    && cd ./modyn/storage/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DMODYNSTORAGE_BUILD_TESTS=0 -DMODYNSTORAGE_BUILD_PLAYGROUND=0 \
    && make -j8 modyn-storage

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ./modyn/storage/build/modyn-storage ./modyn/config/examples/modyn_config.yaml