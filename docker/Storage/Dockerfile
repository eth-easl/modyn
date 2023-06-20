FROM modynbase:latest

ENV CMAKE_VERSION=3.26.4
ARG BUILDPLATFORM

# Copy the shell script
COPY transform_buildplatform.sh /usr/local/bin/transform_buildplatform.sh
RUN chmod +x /usr/local/bin/transform_buildplatform.sh

# Transform BUILDPLATFORM value to aarch64 or x86_64
ENV TRANSFORMED_BUILDPLATFORM=""
RUN TRANSFORMED_BUILDPLATFORM=$(/usr/local/bin/transform_buildplatform.sh $BUILDPLATFORM) \
    && export TRANSFORMED_BUILDPLATFORM

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-Linux-$TRANSFORMED_BUILDPLATFORM.sh \
    -q -O /tmp/cmake-install.sh

RUN chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-$CMAKE_VERSION \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-$CMAKE_VERSION \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-$CMAKE_VERSION/bin/* /usr/local/bin \
    && cmake --version

RUN /opt/conda/bin/conda run -n modyn python -c 'import modyn; print(modyn.__path__[0])' > modynpath.txt && \
    MODYNPATH=`cat modynpath.txt` && \
    mkdir -p $MODYNPATH/storage/build && \
    cd $MODYNPATH/storage/build && cmake .. && make -j8

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml