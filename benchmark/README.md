# Benchmark

Benchmark is a collection of scripts that enable the running of benchmarks on the modyn system.

## Install additional dependencies

To install the additional dependencies necessary to run the scripts in this directory, update the conda environment:

`conda env update --name modyn --file benchmark_environment.yml`

## Benchmarks

### MNIST 
Benchmark Storage Script - Downloads the MNIST dataset into a given directory. For more information on parameters run with `-h`.


### Criteo 1TB Dataset
Readme contains information on how data is downloaded and preprocessed. Current pre processed data has been uploaded to the dds-criteo bucket (gs://dds-criteo/) in the dynamic-datasets google cloud project
