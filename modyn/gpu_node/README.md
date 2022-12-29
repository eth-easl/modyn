# GPU Node objects

This contains the functionality of GPU Node objects (trainer, dataset, metadata collector).

### Workflow

A grpc service runs on the gpu node, and receives requests from the supervisor. The supervisor will:
* register training, which will return a training id, and setup all the necessary structures (model, dataset, etc.)
* start training, which will spawn a new process.

When registering a training configuration, the supervisor sends a grpc request containing:
* model_id and model configuration
* training hyperparameters (batch size, learning rate, etc.) and optimizer parameters
* checkpoint information
* dataset information

For starting a new training session, the supervision needs to provide the training_id and (optionally) a checkpoint to load from.

### OnlineDataset

The 'OnlineDataset' is the base class for user-defined datasets. It maintains two connections, to the storage and selector. It gets keys, i.e. sample ids, from the selector, and retrieves the data from the sample ids.

The user is responsible for defining the transformations done to the dataset. The user should define a subclass of 'OnlineDataset', that, at least overrides the '_process' function, with the correct transformations.
As an example, see the 'data/mnist_dataset.py' file.