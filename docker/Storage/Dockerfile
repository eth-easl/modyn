FROM modyndependencies:latest

COPY ./CMakeLists.txt /src
COPY ./cmake /src/cmake
COPY ./modyn/CMakeLists.txt /src/modyn/CMakeLists.txt
COPY ./modyn/storage /src/modyn/storage
COPY ./modyn/common/CMakeLists.txt /src/modyn/common/CMakeLists.txt
COPY ./modyn/common/cpp /src/modyn/common/cpp
COPY ./modyn/common/example_extension /src/modyn/common/example_extension
COPY ./modyn/common/trigger_sample /src/modyn/common/trigger_sample
COPY ./modyn/protos/storage.proto /src/modyn/protos/storage.proto

COPY ./modyn/selector/internal/storage_backend/local /src/modyn/selector/internal/storage_backend/local
COPY ./modyn/selector/CMakeLists.txt /src/modyn/selector/CMakeLists.txt
COPY ./modyn/selector/internal/CMakeLists.txt /src/modyn/selector/internal/CMakeLists.txt
COPY ./modyn/selector/internal/storage_backend/CMakeLists.txt /src/modyn/selector/internal/storage_backend/CMakeLists.txt

WORKDIR /src
RUN chown -R appuser /src
USER appuser

RUN mkdir build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=${MODYN_BUILDTYPE} -DMODYN_BUILD_TESTS=Off -DMODYN_BUILD_PLAYGROUND=Off -DMODYN_BUILD_STORAGE=On \
    && make -j8 modyn-storage

# These files are copied after building the storage to avoid rebuilding if the config changes
COPY ./modyn/config /src/modyn/config
COPY ./conf /src/conf


# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ./build/modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml
