FROM modynbase:latest

ENV CMAKE_VERSION=3.26.4
ENV CMAKE_DIR /opt/cmake

# Determine the architecture and set it as an environment variable
RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then ARCHITECTURE=aarch64; else ARCHITECTURE=x86_64; fi \
    && wget --quiet "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-${ARCHITECTURE}.sh" \
    -O ~/cmake-install.sh && \
    /bin/bash ~/cmake-install.sh --skip-license && \
    rm ~/cmake-install.sh

RUN ls -la ~/

# Move CMake to an opt folder
RUN mv cmake-${CMAKE_VERSION}-Linux-${ARCHITECTURE} $CMAKE_DIR

RUN ls -la $CMAKE_DIR

ENV PATH=$CMAKE_DIR/bin:$PATH

# Verify CMake installation
RUN cmake --version

RUN mkdir -p ./modyn/storage/build \
    && cd ./modyn/storage/build \
    && cmake .. \
    && make -j8

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ./modyn/storage/build/modyn-storage ./modyn/config/examples/modyn_config.yaml
