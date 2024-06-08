
# Evaluation of models

## Configuration

To evaluate models you can configure several `EvalHandlerConfig` within the `evaluation` section of the 
pipeline configuration file.

There are two main configuration elements:

- evaluation trigger time: Configures when an evaluation should be triggered.
- evaluation strategy: This determines which data should be used for evaluation.

### Evaluation Trigger Time

We support periodic (by sample count or time), static (pre-configured) and AfterEveryTrainingEvalTriggerConfig.

### Evaluation Strategy

#### `IntervalEvalStrategyConfig`

Allows to configure a two sided interval that defines the data used for evaluation.

```mermaid
gantt
    title Evaluation Strategy Interval Configurations
    dateFormat  YYYY-MM-DD
    axisFormat  %Y

    section Training
    -inf : milestone, m2, 2020, 0d
    -2y : milestone, m2, 2021, 0d
    -1y : milestone, m2, 2022, 0d
    -0 : milestone, m1, 2023, 0d
    training interval :active, start, 2023, 2y
    +0 : milestone, m2, 2025, 0d
    1y : milestone, m2, 2026, 0d
    2y : milestone, m2, 2027, 0d
    +inf : milestone, m2, 2028, 0d


    section Examples
    [-inf, -0] :, 2020, 3y
    [-inf, 0] = [-inf, +0] :, 2020, 5y
    [0, +inf] = [-0, +inf] :, 2023, 5y
    [+0, +inf] :, 2025, 3y
    [-1y, +1y] :, 2022, 4y
```
