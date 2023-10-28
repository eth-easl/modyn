FROM modynbase:latest

RUN mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DMODYN_BUILD_TESTS=Off -DMODYN_BUILD_PLAYGROUND=Off -DMODYN_BUILD_STORAGE=On \
    && make -j8 modyn-storage

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ./build/modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml