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
        fork_state --> wait_for_new_data
        state replay_data {
            [*] --> REPLAY_DATA
            REPLAY_DATA_DONE --> [*]
        }
        REPLAY_DATA --> new_data
        new_data --> REPLAY_DATA_DONE
        state wait_for_new_data {
            [*] --> FETCH_NEW_DATA
            FETCH_NEW_DATA --> WAIT_FOR_NEW_DATA
            WAIT_FOR_NEW_DATA --> FETCH_NEW_DATA
            WAIT_FOR_NEW_DATA --> [*]
        }
        state join_state <<join>>
        replay_data --> join_state
        wait_for_new_data --> join_state
        join_state --> DONE
    FETCH_NEW_DATA --> new_data
    new_data --> FETCH_NEW_DATA
    state train_and_eval {
        [*] --> training
        state post_train <<fork>>
            state training {
                [*] --> RUN_TRAINING
                RUN_TRAINING --> TRAINING_COMPLETED
                TRAINING_COMPLETED --> STORE_TRAINED_MODEL
                STORE_TRAINED_MODEL --> [*]
            }
            training --> post_train
            state evaluation {
                [*] --> EVALUATE
                EVALUATE --> EVALUATION_COMPLETED
                EVALUATION_COMPLETED --> STORE_EVALUATION_RESULTS
                STORE_EVALUATION_RESULTS --> end_execute_trigger[*]
            }
            post_train --> evaluation
            post_train --> [*]
        evaluation --> [*]
    }
    state new_data {
        [*] --> HANDLE_NEW_DATA
        HANDLE_NEW_DATA --> new_data_batch
        state new_data_batch {
            [*] --> EVALUATE_TRIGGER_ON_BATCH

            state trigger_decision <<join>>
            EVALUATE_TRIGGER_ON_BATCH --> trigger_decision
            trigger_decision --> INFORM_SELECTOR_NO_TRIGGER
            trigger_decision --> EXECUTE_TRIGGERS_WITHIN_BATCH

            state execute_trigger {
                [*] --> INFORM_SELECTOR_AND_TRIGGER
                INFORM_SELECTOR_AND_TRIGGER --> TRAIN_AND_EVALUATE
                TRAIN_AND_EVALUATE --> [*]
            }
            EXECUTE_TRIGGERS_WITHIN_BATCH --> execute_trigger
            execute_trigger --> INFORM_SELECTOR_REMAINING_DATA

            INFORM_SELECTOR_NO_TRIGGER --> [*]
            INFORM_SELECTOR_REMAINING_DATA --> [*]
        }
        new_data_batch --> NEW_DATA_HANDLED
        NEW_DATA_HANDLED --> [*]
    }

    TRAIN_AND_EVALUATE --> train_and_eval
    train_and_eval --> TRAIN_AND_EVALUATE
    
    DONE --> EXIT
    EXIT --> [*]
```
