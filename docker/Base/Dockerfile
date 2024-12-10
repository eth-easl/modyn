FROM modyndependencies

# Copy source code into container
COPY modyn /src/modyn
COPY cmake /src/cmake
COPY modynclient /src/modynclient
COPY integrationtests /src/integrationtests
COPY setup.py setup.cfg CMakeLists.txt dev-requirements.txt /src/

RUN mamba run -n modyn pip install -e /src
WORKDIR /src
RUN chown -R appuser /src

# Setup environment for user
USER appuser
ENV CONDA_DEFAULT_ENV modyn
ENV MAMBA_DEFAULT_ENV modyn
RUN /bin/bash -c "mamba shell init bash"
RUN echo "mamba activate modyn" >> /home/appuser/.bashrc

# set environment variable to tell modyn that it is running in a container
ENV MODYN_DOCKER 1
