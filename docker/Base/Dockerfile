FROM modyndependencies

ARG MODYN_BUILDTYPE=Release
ENV MODYN_BUILDTYPE=$MODYN_BUILDTYPE

# Copy source code into container
ADD . /src
RUN echo "Used buildtype is ${MODYN_BUILDTYPE}" >> /src/.modyn_buildtype

RUN mamba run -n modyn pip install -e /src
WORKDIR /src
RUN chown -R appuser /src

# Setup environment for user
USER appuser
ENV CONDA_DEFAULT_ENV modyn
ENV MAMBA_DEFAULT_ENV modyn
RUN /bin/bash -c "mamba init"
RUN echo "mamba activate modyn" >> /home/appuser/.bashrc
