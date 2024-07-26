
# Evaluation of models

## Configuration

To evaluate models you can configure several `EvalHandlerConfig` within the `evaluation` section of the 
pipeline configuration file.

### Evaluation Strategy

Evaluation strategies describe how the start and end intervals for evaluations are generated.

#### `_IntervalEvalStrategyConfig`

> base class used for strategies like PeriodicEvalStrategyConfig

Similar to `OffsetEvalStrategy` but allows to configure a two sided interval that defines the data used for evaluation.
`OffsetEvalStrategy` only allows offsets in one direction of the evaluation point `0`.

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
