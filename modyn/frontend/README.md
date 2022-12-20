# GPU Node objects

This contains the functionality of GPU Node objects (trainer, dataset, metadata collector).

### Workflow

The supervisor sends a configuration file containing:
* model (architecture) configuration
* training hyperparameters (batch size, learning rate, epochs, etc.)
* online data processing pipeline
* checkpoint to load from (if any)
* distributed training configuration, e.g. number of workers (could be a subsequent feature)

1. We can have a form of Agent setting up the training. Would this (and basically the GPU resources) busy-wait for triggers from the supervisor? Or will they be stopped and relaunched (resources freed and GPU nodes spawned as requested) when the supervisor signals?
2. Actual training loop may vary from case to case. The users should extend the training loop function.

### Dynamic Dataset

In the case of distributed training, we should take into account the world size, and rank id for queyring the selector.

### Metadata selector

This should talk to the Metadata Processor - TBD
1. Should the metrics be passed as an argument in the configuration file that the supervisor provides?