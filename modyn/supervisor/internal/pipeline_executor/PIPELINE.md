# Pipeline

## Pipeline Orchestration

### Current pipeline

- control flow not very clear
- stage dependencies are not modeled explicitly
- parallel pipeline steps are unclear
- multiple transitions to the same stage (e.g. WAIT_FOR_TRAINING_COMPLETION) while processing sequentially...

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> INIT_CLUSTER_CONNECTION
    INIT_CLUSTER_CONNECTION --> GET_SELECTOR_BATCH_SIZE
    state fork_state <<fork>>
        GET_SELECTOR_BATCH_SIZE --> fork_state
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
                RUN_TRAINING --> WAIT_FOR_TRAINING_COMPLETION
                WAIT_FOR_TRAINING_COMPLETION --> 
                TRAINING_COMPLETED
                TRAINING_COMPLETED --> STORE_TRAINED_MODEL
                STORE_TRAINED_MODEL --> [*]
            }
            training --> post_train
            state evaluation {
                [*] --> EVALUATE
                EVALUATE --> WAIT_FOR_EVALUATION_COMPLETION
                WAIT_FOR_EVALUATION_COMPLETION --> EVALUATION_COMPLETED
                EVALUATION_COMPLETED --> STORE_EVALUATION_RESULTS
                STORE_EVALUATION_RESULTS --> [*]
            }
            post_train --> evaluation
            post_train --> [*]
        evaluation --> [*]
    }
    state new_data {
        [*] --> HANDLE_NEW_DATA
        HANDLE_NEW_DATA --> new_data_batch
        state new_data_batch {
            [*] --> HANDLE_TRIGGERS_WITHIN_BATCH
            HANDLE_TRIGGERS_WITHIN_BATCH --> INFORM_SELECTOR_AND_TRIGGER
            HANDLE_TRIGGERS_WITHIN_BATCH --> INFORM_SELECTOR_REMAINING_DATA
            INFORM_SELECTOR_AND_TRIGGER --> INFORM_SELECTOR_REMAINING_DATA
            INFORM_SELECTOR_REMAINING_DATA --> [*]
        }
        new_data_batch --> NEW_DATA_HANDLED
        NEW_DATA_HANDLED --> [*]
    }

    INFORM_SELECTOR_AND_TRIGGER --> train_and_eval
    train_and_eval --> HANDLE_TRIGGERS_WITHIN_BATCH
    
    DONE --> EXIT
    EXIT --> [*]
```
