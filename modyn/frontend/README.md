# GPU Node objects

This contains the functionality of GPU Node objects (trainer, dataset, metadata collector).

### Workflow

The supervisor sends a grpc request containing:
* model (architecture) configuration
* training hyperparameters (batch size, learning rate, epochs, etc.)
* online data processing pipeline
* checkpoint to load from (if any)

1. A grpc server listens for supervisor requests and spawns training when needed. Training is agnostic to selector policy.
2. A model id is provided by the supervisor. This should correspond to a training function registered by the user.

### Dynamic Dataset

In the case of distributed training, we should take into account the world size, and rank id for queyring the selector.

### Metadata selector

This should talk to the Metadata Processor - TBD
1. Should the metrics be passed as an argument in the configuration file that the supervisor provides?