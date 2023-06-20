FROM modynbase:latest

ENV CMAKE_VERSION=3.26.4

RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then ARCHITECTURE=aarch64; else ARCHITECTURE=x86_64; fi \
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-${ARCHITECTURE}.sh \
    -O ~/cmake-install.sh && \
    /bin/bash ~/cmake-install.sh --skip-license -p /usr/local

RUN mkdir -p ./modyn/storage/build \
    && cd ./modyn/storage/build \
    && conda run -n modyn cmake .. \
    && conda run -n modyn make -j8

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml