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

    # Setup
    INIT = "Initialize pipeline executor"
    INIT_CLUSTER_CONNECTION = "Initialize cluster connection"

    _FORK_DATA_STRATEGY = "Pipeline fork: data strategy"
    """decide whether to choose replay or fetch new data"""

    # Replay Data
    REPLAY_DATA = "Replay data"
    REPLAY_DATA_DONE = "Replay data done"

    # Wait for new data
    FETCH_NEW_DATA = "Fetch new data"
    WAIT_FOR_NEW_DATA = "Wait for new data"

    # Process new data
    HANDLE_NEW_DATA = "Handle new data"

    EVALUATE_TRIGGER_ON_BATCH = "Evaluate trigger on batch"
    INFORM_SELECTOR_NO_TRIGGER = "Inform selector about no trigger"

    EXECUTE_TRIGGERS_WITHIN_BATCH = "Handle triggers within batch"
    INFORM_SELECTOR_AND_TRIGGER = "Inform selector and trigger"

    INFORM_SELECTOR_REMAINING_DATA = "Inform selector about remaining data"

    NEW_DATA_HANDLED = "New data handled"

    # Training
    RUN_TRAINING = "Run training"

    WAIT_FOR_TRAINING_COMPLETION = "Wait for training completion"
    """Implements busy waiting for training completion."""

    TRAINING_COMPLETED = "Training completed 🚀"
    STORE_TRAINED_MODEL = "Store trained model"

    _FORK_DECIDE_EVALUATION = "Pipeline fork: decide evaluation"
    """decide whether to choose evaluate the trained model or not"""

    # Evaluation
    EVALUATE = "Run evaluation"
    WAIT_FOR_EVALUATION_COMPLETION = "Wait for evaluation completion"
    EVALUATION_COMPLETED = "Evaluation completed 🚀"
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
