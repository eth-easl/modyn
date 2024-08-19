# Performance Triggering

## Overview

The `PerformanceTrigger` is a trigger mechanism that evaluates the performance of a model over time and makes decisions based on performance-observing criteria. Unlike simple triggers, the `PerformanceTrigger` considers user selected performance metrics and can evaluate decisions based on static or dynamic performance thresholds, as well as the number of avoidable misclassifications.

We define `avoidable misclassifications` as the misclassifications that could have been avoided if the model had triggered earlier. They can be see as the lost benefits of not triggering. We derive the avoidable misclassification estimates by observing two time series:

1. The series of evaluation scores directly after the last executed triggers (with fresh models). We define those results as `expected performance`.

2. The series of evaluation SINCE the last trigger on policy update points where we could have triggered but decided not to. This series is used to estimate the avoidable misclassifications. We define those results as `actual/effective performance`.

The delta between the expected and the actual performance is then used to estimate the avoidable misclassifications.

This mechanism relies on performance and data density tracking to observe trends. Decision policies then make binary decisions (trigger or not) based on this information.

For the performance and misclassification estimations we can always take two approaches: `hindsight` and `lookahead`. The `hindsight` approach uses the most recent data to estimate the performance and misclassifications. The `lookahead` approach uses the forecasted performance and misclassifications to make decisions.

### Main Architecture

```mermaid
classDiagram
    class Trigger {
        <<abstract>>
        +void init_trigger(TriggerContext context)
        +Generator[Triggers] inform(new_data)
        +void inform_new_model(int most_recent_model_id)
    }

    class PerformanceTrigger {
        +PerformanceTriggerConfig config
        +DataDensityTracker data_density
        +PerformanceTracker performance_tracker
        +dict[str, PerformanceDecisionPolicy] decision_policies

        +void inform_new_model(int most_recent_model_id)


        +void init_trigger(TriggerContext context)
        +Generator[triggers] inform(new_data)
        +void inform_new_model(int most_recent_model_id)
    }

    class PerformanceTriggerConfig {
    }

    class PerformanceDecisionPolicy {
    }

    class DataDensityTracker {
    }

    class PerformanceTracker {
    }

    Trigger <|-- PerformanceTrigger
    PerformanceTrigger *-- "1" PerformanceTriggerConfig
    PerformanceTrigger *-- "1" DataDensityTracker
    PerformanceTrigger *-- "1" PerformanceTracker
    PerformanceTrigger *-- "|decision_policies|" PerformanceDecisionPolicy
```

### `DataDensityTracker` and `PerformanceTracker`

As mentioned, we derive triggering decisions by observing (time-) series of performance metrics and data density and forecasting future values based on past trends. The `DataDensityTracker` is responsible for tracking the density of incoming data. `PerformanceTracker` tracks the model's performance over time. It stores performance metrics after each trigger and forecasts future performance based on recent trends.

```mermaid
classDiagram
    class DataDensityTracker {
        +void inform_data(list[tuple[int, int]])
        +int previous_batch_samples()
        +float forecast_density(method)
    }

    class PerformanceTracker {
        +void inform_evaluation(num_samples, num_misclassifications, evaluation_scores)
        +void inform_trigger(num_samples, num_misclassifications, evaluation_scores)
        +int previous_batch_num_misclassifications()
        +float forecast_expected_accuracy(method)
        +float forecast_next_accuracy(method)
        +float forecast_expected_performance(metric, method)
        +float forecast_next_performance(metric, method)
    }
```

### `PerformanceTriggerCriterion` Hierarchy

The PerformanceTriggerCriterion class is an abstract configuration base class for criteria like `StaticPerformanceThresholdCriterion` and `DynamicPerformanceThresholdCriterion`, which define how decisions are made based on drift metrics. Even if the threshold generation is somewhat different to the DecisionCriteria in the drift triggers, the concept is similar. The configuration made in those models will determine the decision process within the `PerformanceDecisionPolicy` classes.

```mermaid
classDiagram
    class PerformanceTriggerCriterion {
        <<abstract>>
    }

    class PerformanceThresholdCriterion {
        <<abstract>>
        +str metric
    }

    class StaticPerformanceThresholdCriterion {
        +float metric_threshold
    }

    class DynamicPerformanceThresholdCriterion {
        +float allowed_deviation
    }

    class NumberAvoidableMisclassificationCriterion {
        <<abstract>>
    }

    class StaticNumberAvoidableMisclassificationCriterion {
        +Optional[float] expected_accuracy
        +int avoidable_misclassification_threshold
        +bool allow_reduction
    }

    PerformanceThresholdCriterion <|-- StaticPerformanceThresholdCriterion
    PerformanceThresholdCriterion <|-- DynamicPerformanceThresholdCriterion
    PerformanceTriggerCriterion <|-- PerformanceThresholdCriterion
    PerformanceTriggerCriterion <|-- NumberAvoidableMisclassificationCriterion
    NumberAvoidableMisclassificationCriterion <|-- StaticNumberAvoidableMisclassificationCriterion

```

### `PerformanceDecisionPolicy` Hierarchy

The `PerformanceDecisionPolicy` is an abstract base class that defines the decision-making process for performance triggers. It is extended by specific policies that define how decisions are made:

1. **StaticPerformanceThresholdDecisionPolicy**: Triggers when the performance metric falls below a static threshold.
2. **DynamicPerformanceThresholdDecisionPolicy**: Triggers when the performance metric deviates beyond a dynamic threshold, based on a rolling average.
3. **StaticNumberAvoidableMisclassificationDecisionPolicy**: Triggers when the number of avoidable misclassifications exceeds a predefined static threshold.

<details>
<summary><b>What are avoidable misclassification?</b></summary>

Avoidable misclassifications are the misclassifications that could have been avoided if the model had triggered earlier. They can be seen as the lost benefits of not triggering. Every evaluation point we estimate the avoidable misclassifications by comparing the expected performance (the performance of the model if we had triggered) with the actual performance (the performance of the model since the last trigger). Avoidable misclassifications are then derived from the difference between the expected and actual accuracy and the estimated query density.

</details>

<details>
<summary><b>Hindsight vs. Forecasting mode</b></summary>
All decision work on estimates of e.g. the expected performance, next actual/observed performance (if no trigger is executed now) and the estimated query density. The estimates can be made in two modes: `hindsight` and `forecasting`.

In hindsight mode, decisions are based solely on the current observed performance metrics. This mode checks if the most recent performance falls below a predefined threshold or deviates significantly from the expected performance, based on past evaluations. It is a retrospective approach, relying on already observed data to determine if a trigger should be invoked. Triggering happens post-factum when the performance has already degraded / a threshold has been crossed.

On the other hand, forecasting mode looks forward and incorporates predictions of future performance into the decision-making process. This mode not only evaluates the current performance but also forecasts future performance and data density to anticipate potential issues before they occur. For example, in the static number avoidable misclassification policy, forecasting mode predicts whether the cumulative number of avoidable misclassifications will exceed the threshold before the next evaluation. By doing so, it aims to prevent performance degradation by triggering preemptively, rather than waiting for the problem to manifest.

</details>

```mermaid
classDiagram
    class PerformanceDecisionPolicy {
        <<abstract>>
        +bool evaluate_decision(update_interval, evaluation_scores, data_density, performance_tracker, mode, method)*
        +void inform_trigger()
    }

    class StaticPerformanceThresholdDecisionPolicy {
        +StaticPerformanceThresholdCriterion config
        +bool evaluate_decision(...)
    }

    class DynamicPerformanceThresholdDecisionPolicy {
        +DynamicPerformanceThresholdCriterion config
        +bool evaluate_decision(...)
    }

    class StaticNumberAvoidableMisclassificationDecisionPolicy {
        +StaticNumberAvoidableMisclassificationCriterion config
        +bool evaluate_decision(...)
        +void inform_trigger()
    }


    PerformanceDecisionPolicy <|-- StaticPerformanceThresholdDecisionPolicy
    PerformanceDecisionPolicy <|-- DynamicPerformanceThresholdDecisionPolicy
    PerformanceDecisionPolicy <|-- StaticNumberAvoidableMisclassificationDecisionPolicy
```
