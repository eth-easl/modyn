# Cost-Based Triggering

## Overview

Cost-based triggers evaluate the trade-off between the cost of triggering (e.g., training time) and the benefits gained from triggering (e.g., reducing regret metrics like data incorporation latency or avoidable misclassification latency).

As regret, we define metrics that quantify the negative impact of not addressing a problem in a timely manner. They
therefore help in decision making processes like scheduling or retraining decisions.
In the context of `modyn` triggering, we can view regret based metrics as the cost of not triggering.

The `CostTrigger` class serves as the base class for specific implementations, such as `DataIncorporationLatencyCostTrigger` and `AvoidableMisclassificationCostTrigger`, which utilize different regret metrics.

While no trigger occurs regret units are cumulated into to a regret over time curve.
Rather than using this regret units directly, we build an area-under-the-curve metric.
The area under the regret curve measures the time regret units have spent being unaddressed.

As this policy operates the two metrics `time` (cost) and a regret metric we need
a way to express the tradeoff between the two. A user e.g. has to specify how many seconds of training time he is
willing to eradicate a certain amount of cumulative regret.

### Main Architecture

```mermaid
classDiagram
    class Trigger {
        <<abstract>>
        +void init_trigger(trigger_context)
        +Generator[Triggers] inform(new_data)*
        +void inform_new_model(model_id, ...)
    }

    class BatchedTrigger {
        <<abstract>>
        +void inform(...)
        -bool evaluate_batch(batch, trigger_index)*
    }

    class WarmupTrigger {
        -int warmup_intervals
        +delegate_inform(batch)
    }

    class PerformanceTriggerMixin {
        +DataDensityTracker data_density
        +PerformanceTracker performance_tracker
        +StatefulModel model

        -EvaluationResults run_evaluation(interval_data)
    }

    class CostTrigger {
        <<abstract>>
        +CostTracker cost_tracker
        +IncorporationLatencyTracker latency_tracker

        +void init_trigger(trigger_context)
        +void inform_new_model(model_id, ...)

        -bool evaluate_batch(batch, trigger_index)
        -float compute_regret_metric()*

    }

    class CostTracker {
    }

    class IncorporationLatencyTracker {
    }

    class DataIncorporationLatencyCostTrigger {
        -float compute_regret_metric()
    }

    class AvoidableMisclassificationCostTrigger {
        NumberAvoidableMisclassificationEstimator misclassification_estimator

        +void init_trigger(trigger_context)
        +void inform_new_model(model_id, ...)

        -float compute_regret_metric()
    }


    Trigger <|-- BatchedTrigger
    BatchedTrigger <|-- CostTrigger

    BatchedTrigger o-- "0..1" WarmupTrigger
    Trigger "1" --* WarmupTrigger

    PerformanceTriggerMixin <|-- CostTrigger

    CostTrigger *-- "1" CostTracker
    CostTrigger *-- "1" IncorporationLatencyTracker
    CostTrigger <|-- DataIncorporationLatencyCostTrigger
    CostTrigger <|-- AvoidableMisclassificationCostTrigger

    DataIncorporationLatencyCostTrigger *-- "1" DataIncorporationLatencyCostTriggerConfig
    AvoidableMisclassificationCostTrigger *-- "1" AvoidableMisclassificationCostTriggerConfig

    style PerformanceTriggerMixin fill:#DDDDDD,stroke:#A9A9A9,stroke-width:2px

    style DataIncorporationLatencyCostTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
    style AvoidableMisclassificationCostTrigger fill:#C5DEB8,stroke:#A9A9A9,stroke-width:2px
```

### `CostTrigger` Hierarchy

Both `DataIncorporationLatencyCostTrigger` and `AvoidableMisclassificationCostTrigger` track the cost of triggering and convert a regret metric (e.g., data incorporation latency or avoidable misclassification latency) into the training time unit with a user-defined conversion factor.

<details>
<summary><b>Incorporation Latency</b></summary>

Incorporation latency measures the delay in integrating / addressing new data or drift problems. They are typically set up as a area-under-the-curve metric, where the area is the time taken to incorporate the data. The underlying curve function is the number of samples or regret units over time that need to be addressed.

</details>

#### `DataIncorporationLatencyCostTrigger`

- Uses data incorporation latency as the regret metric
- Measures the delay in integrating new data
- Triggers when the accumulated integration delay converted to time units (user defined conversion factor) exceed the expected training time

#### `AvoidableMisclassificationCostTrigger`

- Extends performance-aware triggers with a cost-awareness aspect
- Uses avoidable misclassification latency as the regret metric
- Triggers when the accumulated misclassification latency (in training time units) exceeds the expected training time.

### `CostTracker`

- **Purpose**: Tracks and forecasts the cost of triggers (e.g., wall clock time) based on past trigger data.
- **Functionality**:
  - Records the number of samples processed and the time taken for each trigger.
  - Uses a linear regression model to forecast future training times based on the number of samples.
  - Requires calibration after each trigger to refine its predictions.

```mermaid
classDiagram
    class CostTracker {
        +deque measurements
        +Ridge linear_model
        +void inform_trigger(int number_samples, float elapsed_time)
        +bool needs_calibration()
        +float forecast_training_time(int number_samples)
    }
```

### `IncorporationLatencyTracker`

- **Purpose**: Tracks latency-based regret metrics like data-incorporation-latency.
- **Functionality**:
  - Stores how many units of regret have been seen so far, and their latency.
  - The regret latency measures the number of seconds regret units have spent being unaddressed by a trigger.
  - When modeling a regret over time curve, the current regret is value of the curve at the current time.
  - The cumulative latency regret is the area under the curve up to the current time.

```mermaid
classDiagram
    class IncorporationLatencyTracker {
        +float current_regret
        +float cumulative_latency_regret

        +float add_latency(float regret, float batch_duration)
        +float add_latencies(list[tuple[int, float]] regrets, int start_timestamp, float batch_duration)
        +void inform_trigger()
    }
```

### `CostTriggerConfig`

```mermaid
classDiagram
    class CostTriggerConfig {
        +int cost_tracking_window_size
    }

    class DataIncorporationLatencyCostTriggerConfig {
        +float incorporation_delay_per_training_second
    }

    class PerformanceTriggerConfig {
    }

    class AvoidableMisclassificationCostTriggerConfig {
        +float avoidable_misclassification_latency_per_training_second
    }

    CostTriggerConfig <|-- DataIncorporationLatencyCostTriggerConfig
    CostTriggerConfig <|-- AvoidableMisclassificationCostTriggerConfig
    PerformanceTriggerConfig <|-- AvoidableMisclassificationCostTriggerConfig

```
