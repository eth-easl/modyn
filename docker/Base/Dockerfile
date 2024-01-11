FROM modyndependencies

# Copy source code into container
COPY . /src

RUN mamba run -n modyn pip install -e /src
WORKDIR /src
RUN chown -R appuser /src

# Setup environment for user
USER appuser
ENV CONDA_DEFAULT_ENV modyn
ENV MAMBA_DEFAULT_ENV modyn
RUN /bin/bash -c "mamba init"
RUN echo "mamba activate modyn" >> /home/appuser/.bashrc
