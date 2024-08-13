# Pipeline Logging

## Motivation

We want to be able to log the progress of the pipeline, so that we can conveniently analyze experiments and compare
different pipeline configurations.

## Design

For that purpose, we are measuring the execution time of every pipeline step function call.
Note that every pipeline stage corresponds to one function in `pipeline_executor.py`.

There are 4 main pydantic models in `models.py` that are used to create pipeline execution logs. We are using pydantic as it provides
a fully typed and validated data structure that can easily be serialized and deserialized into json files.

### `PipelineLogs`

This is the main model that contains all the logs for a single pipeline execution. One `PipelineLogs` will produce one
json file that contains all the logs for a single pipeline execution.

### `SupervisorLogs`

One member of `PipelineLogs` is a `SupervisorLog`. This model contains logs produced by / relevant to the
supervisors. Semantically it is a wrapper for a list of `StageLog` entries and provides additional utility functions
for conversion to pandas dataframes.

### `StageLog`

A `StageLog` contains the logs for a specific execution of a specific stage of the pipeline. It records information
like the pipeline stage id, `duration`, `sample_idx`, `sample_time` etc. to able to analyze the pipeline execution quantitatively over time.

These properties are available for every pipeline stage and can therefore be summarized in a pandas dataframe.

Alongside these shared properties some pipeline stages might want to log additional information.
E.g. the results of certain grpc calls to other services like the results of a model evaluation, etc.
This additional optional information can be stored in the `info` field of the `StageLog` which requires an object
from a subtype of `StageInfo`.

### `StageInfo`

Stage info is an arbitrary serializable object that can be stored in the `info` field of a `StageLog`.
Every pipeline stage that wants to log additional information can define its own `StageInfo` subtype.
Additionally a dataframe conversion function can be implemented in the subtype to allow for easy conversion
of the logs of this stage to a pandas dataframe.

## Usage

### Write new information into logs

1. Define a subclass of `StageInfo` in `models.py` that contains the information you want to log.

E.g.

```python
class MyPipelineStageInfo(StageInfo):
    my_data: int = Field(..., description="Some measurement.")

    @override
    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame([(self.my_data)], columns=["my_data"])
```

2. If not already present, define your `pipeline_executor` stage function in `pipeline_executor.py`.

E.g.

```python
    @pipeline_stage(PipelineType.<StageName>, PipelineStage.<parent_stage>)
    def _my_pipeline_stage(
        self, s: ExecutionState, log: StageLog, <arguments from the caller>
    ) -> None:
        <Business logic>

        # Add log information
        log.info = MyPipelineStageInfo(my_data=42)
```

This is sufficient to persist the data to the logfile.

### Read & analyze logs

```python
from modyn.supervisor.internal.pipeline_executor.models import PipelineLogs

# Read the logs from a file
with open("path/to/logfile.json", "r") as f:
    logs = PipelineLogs.model_validate_json(f.read())

# Access the logs
print(logs.supervisor_logs.df)
print(logs.supervisor_logs.df['id'].unique())
print(logs.supervisor_logs.df.describe())

# access the logs of a specific stage (with StageInfo data)
stage_logs = [l for l in logs.supervisor_logs.stage_runs if l.id == "STAGE_NAME"]
stage_df = pd.concat([l.df(extended=True) for l in stage_logs])
stage_df.describe()
```
