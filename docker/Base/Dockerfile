FROM modyndependencies

# Copy source code into container
ADD . /src
RUN conda run -n modyn pip install -e /src
WORKDIR /src
RUN chown -R appuser /src

# Setup environment for user
USER appuser
ENV CONDA_DEFAULT_ENV modyn
RUN /bin/bash -c "conda init"
RUN echo "conda activate modyn" >> /home/appuser/.bashrc
