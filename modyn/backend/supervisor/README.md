# Training Supervisor

This is the Training Supervisor submodule. 

The training supervisor is the brain of Modyn. ML engineers start the supervisor using `./modyn-supervisor pipeline.yaml config.yaml`″.
The first configuration file describes the pipeline setup, while the second configuration file describes the system setup.

## How a pipeline is configured

- Model Type
- Framework (Pytorch/Tensorflow)
- Training setup
    - single GPU
    - multi GPU (with the various suboptions)
    - num workers
    - augmentation pipeline
    - epochs
    - lr / lr scheduling / ...
    - batch size
    - model config, if applicable
- Dataset source and identifier
    - Test set/validation set? needs to be updated over time as well for distribution shift tasks!
- Trigger
    - Time Trigger: Interval
    - Data Count Trigger: How much new data for retraining?
    - Data Shift Trigger: Detection Algorithm
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
- Deployment
    - where to send trained model
- experiment mode
    - replay only: requires initial_mode = replay, does not register trigger on new data but just replays data and ends after replay

## What happens

1. Supervisor validates config
    - Is the model known to the system (implementation exists within the system for PyTorch/Tensorflow)
    - Is everything else valid / implemented in the config yaml?

2. Supervisor validates system
    - GPU nodes available
    - Can we reach the source/storage system and actually find the data set?

3. Register pull OR push (TBD) notification for new data

4. If applicable: Run initial pass

5. Repeat on trigger:
    1. Trigger training 

... Wait for termination (CTRL+C -> deregister training etc)