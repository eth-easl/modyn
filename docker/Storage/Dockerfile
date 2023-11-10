FROM modyndependencies:latest

COPY CMakeLists.txt /src
COPY modyn/storage /src/storage
WORKDIR /src/storage
RUN chown -R appuser /src
USER appuser

RUN mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=${MODYN_BUILDTYPE} -DMODYN_BUILD_TESTS=Off -DMODYN_BUILD_PLAYGROUND=Off -DMODYN_BUILD_STORAGE=On \
    && make -j8 modyn-storage

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ./build/modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml