# Pipeline

## Pipeline Orchestration

### Current pipeline

```mermaid
stateDiagram-v2
    direction LR
    [*] --> INIT
    INIT --> INIT_CLUSTER_CONNECTION
    state fork_state <<fork>>
        INIT_CLUSTER_CONNECTION --> fork_state
        fork_state --> replay_data
        fork_state --> serve_online_data
        state replay_data {
            [*] --> REPLAY_DATA
            REPLAY_DATA --> [*]
        }
        REPLAY_DATA --> process_new_data
        process_new_data --> REPLAY_DATA
        state serve_online_data {
            [*] --> FETCH_NEW_DATA
            FETCH_NEW_DATA --> WAIT_FOR_NEW_DATA
            WAIT_FOR_NEW_DATA --> FETCH_NEW_DATA
            WAIT_FOR_NEW_DATA --> [*]
        }
    replay_data --> DONE
    serve_online_data --> DONE
    state process_new_data {
        state process_new_data_batch {
            [*] --> EVALUATE_TRIGGER_POLICIES
            EVALUATE_TRIGGER_POLICIES --> EXECUTE_TRIGGERS
            EXECUTE_TRIGGERS --> INFORM_SELECTOR_REMAINING_DATA
            INFORM_SELECTOR_REMAINING_DATA --> [*]
        }
        [*] --> process_new_data_batch
        process_new_data_batch --> [*]
    }
    state execute_single_trigger {
        direction LR
        [*] --> INFORM_SELECTOR_AND_TRIGGER
        INFORM_SELECTOR_AND_TRIGGER --> train_and_store_model
        state train_and_store_model {
            [*] --> TRAIN
            TRAIN --> TRAINING_COMPLETED
            TRAINING_COMPLETED --> STORE_TRAINED_MODEL
            STORE_TRAINED_MODEL --> [*]
        }
        state evaluation {
            [*] --> EVALUATE
            EVALUATE --> single_evaluation
            state single_evaluation {
                direction LR
                [*] --> EVALUATE_SINGLE
                EVALUATE_SINGLE --> STORE_EVALUATION_RESULTS
                STORE_EVALUATION_RESULTS -->[*]
            }
            single_evaluation -->  [*]
        }
        train_and_store_model --> evaluation
    }

    EXECUTE_TRIGGERS --> execute_single_trigger
    execute_single_trigger --> EXECUTE_TRIGGERS
FETCH_NEW_DATA --> process_new_data
    process_new_data --> FETCH_NEW_DATA
    DONE --> EXIT
    EXIT --> [*]
```
