# Training Supervisor

This is the Training Supervisor submodule. 

The training supervisor is the brain of Modyn. ML engineers start the supervisor using `./modyn-pipeline pipeline.yaml` 

## How a pipeline is configured

- Model Type
- Framework (Pytorch/Tensorflow)
- Training setup
    - single GPU
    - multi GPU (with the various suboptions)
    - num workers
    - augmentation pipeline
- Dataset source and identifier
    - Test set/validation set? needs to be updated over time as well for distribution shift tasks!
- Trigger
    - training frequency
- Initial Model
    - random initialized
    - pretrained
- The training strategy 
    - Our baselines, GDumb, ...
- Do we do an initial pass to train on all existing data or not?
    - If not, do we replay the existing data as a stream for our training strategy (requires that strategy is not retraining)
    - maybe: `initial_mode = [train_on_everything, replay, ignore]`
- Logging
    - what to log, where to log
- Evaluation tasks?
- experiment mode
    - replay only: requires initial_mode = replay, does not register trigger on new data but just replays data and ends after replay

## What happens

1. Supervisor validates config
    - Is the model known to the system (implementation exists within the system for PyTorch/Tensorflow)

2. Supervisor validates system
    - GPU nodes available
    - Can we reach the source/storage system and actually find the data set?

3. Register pull OR push (TBD) notification for new data

4. T

... Wait for termination (CTRL+C -> deregister training etc)