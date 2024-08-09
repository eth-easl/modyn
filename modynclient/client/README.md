# Modyn Client

This is the modynclient module

The client can connect to the modyn supervisor at the ip and port specified in the modyn_client_config file. ML engineers start a pipeline using `modyn-client pipeline.yaml config.yaml`.
The first configuration file describes the pipeline setup, while the second configuration file describes the modyn system setup.

The evaluation logs will be stored in supervisor's container, in path `modyn_config['supervisor']['eval_directory']`, see [the modyn config file](modyn/config/examples/modyn_config.yaml).
Note that this is only a temporary solution. In future the logs should be lively transferred to client via, e.g., FTP.

Optional command line flag: `--start-replay-at TIMESTAMP`.
This mode does not trigger on new data but just replays data starting at `TIMESTAMP` and ends all training afterwards.
`TIMESTAMP` can be 0 and then just replays all data.
In case of `initial_mode == train_until` with `now` as timestamp or a timestamp that is higher than the replay timestamp in the pipeline config we fail because then the initialization will conflict with the experiment.
We need to think about how communication will work in this case, probably the supervisor will need to tell the storage to replay the data in order to ensure correct communication with the selector etc. TBD

## How a pipeline is configured

- Model Type (ResNet, ...)
- Framework (Pytorch/Tensorflow)
- Training setup
  - single GPU
  - multi GPU (with the various suboptions, parameter server, all reduce, ...)
  - num workers
  - augmentation pipeline
  - lr / lr scheduling / ... (TODO think about the semantics of this in dynamic settings)
  - batch size
  - model config, if applicable, such as number of channels etc, passed directly to model.
- Dataset source and identifier
  - Unclear: How to define test set/validation set? needs to be updated over time as well, at least in some cases where we observe distribution shifts!
- Trigger
  - Time Trigger: Interval
  - Data Count Trigger: How many new data points for retraining?
  - Data Shift Trigger: Detection Algorithm
- Initial Model
  - Randomly initialized
  - Pretrained
- The training strategy
  - Baselines, GDumb, ... with strategy subconfig passed directly to strategy, if applicable.
- Do we do an initial pass to train on all existing data or not?
  - If not, do we replay the existing data as a stream for our training strategy (requires that strategy is not retraining)
  - maybe: `initial_mode = [replay, ignore, train_until]` where train_until expects a subconfig that is either a timestamp or `now` and tells on which data we should train initially. replay = use algorithm for all data, train_until = just train until that data
- Logging
  - what to log, where to log
- Evaluation tasks?
- Deployment
  - where to send trained model

## What happens

1. Client sends a StartPipeline request to the Supervisor

2. Supervisor validates the config files and the system

3. Supervisor starts a pipeline executor process to run the pipeline

4. Client periodically sends a GetPipelineStatus request to the Supervisor and waits until the pipeline terminates
