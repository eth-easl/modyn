FROM modyndependencies:latest
# We rely on modyndependencies so we only need to re-install apex when the dependencies change, not when the source code of Modyn changes

# TODO(#104): Make this easily configurable here
RUN mamba run -n modyn pip install packaging ninja
RUN git clone https://github.com/NVIDIA/apex ./apex
RUN mamba run -v -n modyn pip install -v --no-build-isolation --no-cache-dir --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
