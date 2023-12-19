from enum import Enum


class PipelineStatus(str, Enum):
    RUNNING = "running"
    EXIT = "exit"
    NOTFOUND = "not found"


class PipelineStage(str, Enum):
    INIT = "Initialize pipeline executor"
    INIT_CLUSTER_CONNECTION = "Initialize cluster connection"
    GET_SELECTOR_BATCH_SIZE = "Get selector batch size"
    HANDLE_NEW_DATA = "Handle new data"
    NEW_DATA_HANDLED = "New data handled"
    RUN_TRAINING = "Run training"
    WAIT_FOR_TRAINING_COMPLETION = "Wait for training completion"
    TRAINING_COMPLETED = "Training completed 🚀"
    WAIT_FOR_EVALUATION_COMPLETION = "Wait for evaluation completion"
    EVALUATION_COMPLETED = "Evaluation completed 🚀"
    STORE_TRAINED_MODEL = "Store trained model"
    EVALUATE = "Run evaluation"
    STORE_EVALUATION_RESULTS = "Store evaluation results"
    HANDLE_TRIGGERS_WITHIN_BATCH = "Handle triggers within batch"
    INFORM_SELECTOR_AND_TRIGGER = "Inform selector and trigger"
    INFORM_SELECTOR_REMAINING_DATA = "Inform selector about remaining data"
    REPLAY_DATA = "Replay data"
    REPLAY_DATA_DONE = "Replay data done"
    FETCH_NEW_DATA = "Fetch new data"
    WAIT_FOR_NEW_DATA = "Wait for new data"
    DONE = "Pipeline done"
    EXIT = "Exit"


class MsgType(str, Enum):
    GENERAL = "general_msg"
    DATASET = "dataset_msg"
    COUNTER = "counter_msg"
    ID = "id_msg"
    EXIT = "exit_msg"


class IdType(str, Enum):
    TRIGGER = "trigger"
    TRAINING = "training"


class CounterAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    CLOSE = "close"
