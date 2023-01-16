# Benchmark

In this directory, you find different scripts and examples that assist in running benchmarks on Modyn.
Currently, we provide an MNIST example and a Criteo 1TB Advertising dataset example 

## Install additional dependencies

To install the additional dependencies necessary to run the scripts in this directory, update the conda environment:

`conda env update --name modyn --file benchmark_environment.yml`

## Benchmarks

### MNIST 
Benchmark Storage Script - Downloads the MNIST dataset into a given directory. For more information on parameters run with `-h`.


### Criteo 1TB Dataset
Readme contains information on how data is downloaded and preprocessed. Current pre processed data has been uploaded to the dds-criteo bucket (gs://dds-criteo/) in the dynamic-datasets google cloud project
