# MNIST Example

In this directory, you find files necessary to run experiments on an artificially dynamic version of the MNIST data set.

## Data Generation
You can use the `data_generation.py`script to generate the data.
Use the `-h` flag to find out more.

## Pipeline
In the `mnist.yaml` file, we provide an example MNIST pipeline that can be submitted to the supervisor.
On a system where the supevisor is running, execute `mnist-supervisor --start-replay-at 0 ./mnist.yaml <Modyn Config>` to run the pipeline.
For more information on how to get Modyn running, check out the README in the project root directory.