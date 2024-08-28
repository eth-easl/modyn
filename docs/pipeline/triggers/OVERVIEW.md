# Modyn Triggering

Alongside the simple triggers, Modyn also provides complex triggers that can be used to trigger the training of a model.

Despite the line being blurry, complex triggers are generally more sophisticated and require more information to be provided to the trigger - they most of the time cannot be pre-determined via a simple configuration entry but require
reingested information from the ongoing pipeline run.

Some complex triggers can only make decisions in batched intervals as they are cannot be efficiently computed
on a per-sample basis. Here we can find the `DataDrift` and `Cost` based triggers.

Another policy type is the `EnsemblePolicy` which can be used to combine multiple triggers into a single trigger. This can be useful if multiple triggers should be evaluated before the training of a model is triggered.
One can either use pre-defined ensemble strategies like `MajorityVote` and `AtLeastNEnsembleStrategy` or define custom functions that reduce a list of trigger decisions (one per sub-policy) and make decisions on them freely via the CustomEnsembleStrategy.

```mermaid
classDiagram
    class Trigger {
        <<Abstract>>
        +init_trigger(trigger_context)
        +inform(new_data)*
        +inform_new_model(model_id, num_samples, training_time)
    }

    namespace simple_triggers {

        class TimeTrigger {
        }

        class DataAmountTrigger {
        }

    }

    namespace complex_triggers {

        class BatchedTrigger {
            <<Abstract>>
            +inform(...)
            -bool evaluate_batch(batch, trigger_index)*
        }

        class PerformanceTriggerMixin {
            -EvaluationResults run_evaluation()
        }

        class DataDriftTrigger {
            -bool evaluate_batch(batch, trigger_index)
        }

        class PerformanceTrigger {
            -bool evaluate_batch(batch, trigger_index)
        }

        class CostTrigger {
            <<Abstract>>
            -bool evaluate_batch(batch, trigger_index)
            -float compute_regret_metric()*
        }

        class DataIncorporationLatencyCostTrigger {
            -float compute_regret_metric()
        }

        class AvoidableMisclassificationCostTrigger {
            -float compute_regret_metric()
        }

        class EnsembleTrigger {
        }

    }

    Trigger <|-- BatchedTrigger
    Trigger <|-- EnsembleTrigger

    Trigger <|-- TimeTrigger
    Trigger <|-- DataAmountTrigger

    EnsembleTrigger *-- "n" Trigger

    BatchedTrigger <|-- DataDriftTrigger
    BatchedTrigger <|-- PerformanceTrigger
    BatchedTrigger <|-- CostTrigger

    CostTrigger <|-- DataIncorporationLatencyCostTrigger
    CostTrigger <|-- AvoidableMisclassificationCostTrigger

    PerformanceTriggerMixin <|-- PerformanceTrigger
    PerformanceTriggerMixin <|-- AvoidableMisclassificationCostTrigger

    style PerformanceTriggerMixin fill:#DDDDDD,stroke:#A9A9A9,stroke-width:2px

    style TimeTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
    style DataAmountTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px

    style DataDriftTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
    style PerformanceTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
    style DataIncorporationLatencyCostTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
    style AvoidableMisclassificationCostTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px

    style EnsembleTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
```
