from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class PipelineStatus(StrEnum):
    RUNNING = "running"
    EXIT = "exit"
    NOTFOUND = "not found"


class PipelineStage(StrEnum):
    """For a state transition graph checkout the `PIPELINE.md` file."""

    MAIN = "Pipeline entrypoint"

    # Setup
    INIT = "Initialize pipeline executor"
    INIT_CLUSTER_CONNECTION = "Initialize cluster connection"

    # Replay Data
    REPLAY_DATA = "Replay data"
    REPLAY_DATA_DONE = "Replay data done"

    # Wait for new data
    SERVE_ONLINE_DATA = "Serve online data"
    FETCH_NEW_DATA = "Fetch new data"
    WAIT_FOR_NEW_DATA = "Wait for new data"

    # Process new data
    PROCESS_NEW_DATA = "Process new data"

    PROCESS_NEW_DATA_BATCH = "Process new data batch"
    EVALUATE_TRIGGER_POLICY = "Evaluate trigger on batch"
    INFORM_SELECTOR_NO_TRIGGER = "Inform selector about no trigger"

    # Trigger
    HANDLE_TRIGGERS = "Handle triggers within batch"
    HANDLE_SINGLE_TRIGGER = "Handle single trigger"
    INFORM_SELECTOR_ABOUT_TRIGGER = "Inform selector and trigger"
    INFORM_SELECTOR_REMAINING_DATA = "Inform selector about remaining data"

    NEW_DATA_HANDLED = "New data handled"

    # Training
    TRAIN_AND_STORE_MODEL = "Train and store model"
    TRAIN = "Run training"

    WAIT_FOR_TRAINING_COMPLETION = "Wait for training completion"
    """Implements busy waiting for training completion."""

    TRAINING_COMPLETED = "Training completed 🚀"
    STORE_TRAINED_MODEL = "Store trained model"

    # Evaluation
    EVALUATE = "Run evaluation"
    EVALUATE_SINGLE = "Run single evaluation"
    WAIT_FOR_EVALUATION_COMPLETION = "Wait for evaluation completion"
    STORE_EVALUATION_RESULTS = "Store evaluation results"

    # Teardown
    DONE = "Pipeline done"
    EXIT = "Exit"


class MsgType(StrEnum):
    GENERAL = "general_msg"
    DATASET = "dataset_msg"
    COUNTER = "counter_msg"
    ID = "id_msg"
    EXIT = "exit_msg"


class IdType(StrEnum):
    TRIGGER = "trigger"
    TRAINING = "training"


class CounterAction(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    CLOSE = "close"
