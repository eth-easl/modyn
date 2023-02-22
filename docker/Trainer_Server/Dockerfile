FROM modynbase:latest

RUN conda run -n modyn pip install packaging
RUN git clone https://github.com/NVIDIA/apex ./apex
ENV CPLUS_INCLUDE_PATH=/home/appuser/.conda/pkgs/cuda-nvprof-11.7.101-0/include/
RUN conda run -v -n modyn pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
RUN chmod a+x /src/modyn/trainer_server/modyn-trainer-server

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn --no-capture-output ./modyn/trainer_server/modyn-trainer-server ./modyn/config/examples/modyn_config.yaml
