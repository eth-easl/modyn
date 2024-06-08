# Pipeline

## Pipeline Orchestration

### Current pipeline

```mermaid
stateDiagram-v2
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
            state execute_single_trigger {
                [*] --> INFORM_SELECTOR_AND_TRIGGER
                INFORM_SELECTOR_AND_TRIGGER --> training
                state train_and_store_model {
                    [*] --> TRAIN
                    TRAIN --> TRAINING_COMPLETED
                    TRAINING_COMPLETED --> STORE_TRAINED_MODEL
                    STORE_TRAINED_MODEL --> [*]
                }
                state evaluation {
                    [*] --> EVALUATE
                    EVALUATE --> EVALUATE_SINGLE
                    EVALUATE_SINGLE -->  [*]
                }
                train_and_store_model --> evaluation
            }
            EXECUTE_TRIGGERS --> execute_single_trigger
            execute_single_trigger --> INFORM_SELECTOR_REMAINING_DATA

            INFORM_SELECTOR_REMAINING_DATA --> [*]
        }
        [*] --> process_new_data_batch
        process_new_data_batch --> [*]
    }
    FETCH_NEW_DATA --> process_new_data
    process_new_data --> FETCH_NEW_DATA
    DONE --> EXIT
    EXIT --> [*]
```
