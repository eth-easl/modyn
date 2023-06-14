FROM modynbase:latest

RUN /opt/conda/bin/conda run -n modyn python -c 'import modyn; print(modyn.__path__[0])' > modynpath.txt && \
    MODYNPATH=`cat modynpath.txt` && \
    mkdir -p $MODYNPATH/storage/build && \
    cd $MODYNPATH/storage/build && cmake .. && make -j8

RUN chmod a+x /src/modyn/storage/modyn-storage

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD conda run -n modyn ./modyn/storage/modyn-storage ./modyn/config/examples/modyn_config.yaml