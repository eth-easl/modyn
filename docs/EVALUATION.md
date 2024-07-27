# Modyn pevaluation framework

## 1. Motivation

### 1.1 Use cases

We want to be able to:

1) determine how well our models are performing according to some metrics like accuracy, precision, recall, F1-score, etc.

2) determine how well our models are performing over time. This allows us to determine the generalization and retention capabilities of our models.

3) analyze and compare entire pipeline runs (and not only the different models created over one pipeline run)

4) compare the performance of different pipeline runs with the costs they incurred

We are facing two use cases that require us to run evaluations:

1) **Online evaluations**: We want to run evaluations during the execution of the core pipeline so we can feed back this information into performance-aware triggering policies. For example, we may want to trigger a policy that will retrain a model if the model's performance degrades below a certain threshold.

2) **Post-hoc offline analysis**: We want to run evaluations for experiments so we determine the performance of a model **at certain points in time (3.2.1)** over the course of a whole pipeline run (also for intervals where it has long been replaced with a newer model). Orthogonally to this  we are interested in the **total pipeline performance (3.2.2)** aggregated over the whole time of the pipeline run (= pipeline performance). For this we are only interested in models trained shortly before each evaluation point as this is the model that a production system would use for inference queries.

## 2. Background

### 2.1 Setting

To visualize the different evaluation concepts and strategies, let's consider a completed pipeline run that produced multiple triggers and therefore multiple models. We want to evaluate the performance of each model on a set of evaluation metrics (accuracy, F1, ...).

```mermaid
gantt
    title Pipeline run
    dateFormat YYYY-MM-DD
    axisFormat %Y
    section models
        Trigger1 :milestone, 1930, 0d
        model 1 active          :a1, 1930, 10y
        Trigger2 :milestone, :after a1
        model 2 active          :a2, after a1, 5y
        Trigger3 :milestone, :after a2
        model 3 active          :a3, after a2, 5y
        Trigger4 :milestone, :after a3
        model 4 active          :a4, after a3, 25y
        Trigger5 :milestone, :after a4
        model 5 active          :a5, after a4, 25y
        Trigger6 :milestone, :after a5
        model 6 active          :a6, after a5, 5y
        Trigger7 :milestone, :after a6
        model 7 active          :a7, after a6, 5y
```

### 2.3 Evaluation methods

One evaluation method consists of three main aspects:

1) A set of evaluation metrics (e.g. F1, accuracy, precision, recall, etc.)
2) The points in time when the evaluations are run
3) The evaluation data for the different evaluation triggers (set of samples that are used for an evaluation). Note that this data can be in form of pre-determined dataset slices, or dynamically determined based on the e.g. time of evaluation in form of an interval around the evaluation time.

## 3. Implementation

### 3.1 Online evaluations

#### Runtime considerations

In this case we have to account for the time we spend evaluating as it can be considered part of the core pipeline. Evaluation costs can e.g. be modeled as parts of the trigger policy evaluation costs - or at least part of the core pipeline that cannot be disregarded like the Post-hoc offline analysis (for experiment mode).

During the core pipeline execution, we cannot access future data for evaluation as a production system also wouldn't be able to do that. It might make sense to allow modyn to build a holdout dataset that is isn't used for training but only for evaluations. A first simplifying assumption would be to only evaluate on new data since the last training run that has not been seen by the training server.

Similar windowing strategies as in the post-hoc offline analysis might yield useful results. These windows can limit the time range and sample count of evaluation data.

### 3.2 Post-hoc offline analysis

In experiment mode we can potentially utilize the full evaluation split over the whole dataset time range. Being able to "look ahead" is important to determine the performance of a model post factum as the usage period of model is after the training.

#### Which dimensions are there? (controllable & observable)

Naturally when running a pipeline we can a observe $m$ models being trained over time at the triggers $t_1, ..., t_i \in T$. We refer to $m_i$ as the $i$-th model trained after trigger $t_i$.

Every of these $m$ models can be evaluated at any point in time (bounded by the dataset time range). As we are potentially interested in more than one evaluation metric $q_i \in Q$ ($Q$ being the set of evaluation metrics that are of interest to us), we can evaluate every model at every point in time with every evaluation metric. We therefore get a time series of evaluation results from $M \times Q$.

#### 3.2.1 Analysis of a single pipeline run (comparing the different models)

<details>
  <summary><b>Plot 1: (line plot, x=time, y=eval metric, color=model)
</b></summary>

When analyzing how a single pipeline run performed over time, we can line-plot the evaluation results over time using time as the $x$ dimension, the evaluation metric value as the $y$ dimension and the model as a color coded dimension. The different evaluation metrics can be plotted in different plots using facets.

Using this plot we expect to see that every model performs best at the interval it was trained in(, and hopefully also reasonably well in the interval the model was served in). Accuracy should decay in both directions from the training interval.

There are two main approaches to acquire those time series of evaluation results:
</details>

##### Dataset slicing (`MatrixEvaluationStrategy`)

Partitioning the dataset(s) w.r.t. to time at sufficiently small intervals will allow us to extract a time series after running evaluations for every (slice, model, metric) combination.

When plotting this there are several cases:

- When our slices already correspond to the minimal resolution of the samples times (e.g. every year), we don't need to think about interpolation and can just connect the measurement points with lines.

- When we have more continuous data where we cannot evaluate at every point in time, can either plot vertical lines (same value for a certain time range) or interpolate. In the latter case it's fair to plot the points for each evaluation interval at the interval center and connect the points with lines (linear interpolation).

##### Tumbling window

A simple approach to evaluate models at different points in time is to use a tumbling window. We can slice the dataset into partitions that can e.g. correspond to triggering intervals. Each of the partitions can then be evaluated for every
model and evaluation metric.

##### Sliding windows

(centered around points in time where we intend to query evaluation metrics)

We want to address the problem of not being able to capture the full time resolution of the dataset when slicing it into partitions. We cannot shrink the slices arbitrarily as this would decrease the number of samples in every evaluation split and therefore the reliability of the evaluation results. The reason for the shrinkage in samples when reducing the slice size is that slicing use non intersecting tumbling windows.

We can therefore use sliding windows. For that we modify the understanding of what `Performance at a certain point in time` is. We now not only use the data exactly at that point in time / in fixed non-intersecting partitions but within a sliding window centered around a point. When the window is set big enough, this gives us enough data to have meaningful accuracy results for every point in time we wish to evaluate. Theoretically it allows to assign evaluation metrics to every step in time. This is enabled by the fact that the intervals used for evaluation can overlap here. We can therefore evaluate the model at every point with slightly shifted evaluation data. We definitely don't need to do this at sample level in our use cases but it gives us more flexibility in how we want to evaluate the models. Especially for sparsely filled datasets this can help to regularize away outliers in sparse time-series data. Generally we expect smoother heatmap evaluation matrix plots when using bigger sliding windows.

#### 3.2.2 Comparison of a multiple pipeline runs (comparing the different models)

When not looking at a single pipeline run but at multiple pipeline runs, we gain another dimension that corresponds to the pipeline run.
Consequently, we are dealing with the following dimensions:

- time
- evaluation metric (categorical)
- evaluation metric value
- model
- pipeline run

From now on we will assume the different categorical evaluation metrics will be plotted in different plots, so we consider this dimension fixed now.

What is more, the model dimension is not of any interest anymore here, as we compare at pipeline run level.

That leaves us with the following dimensions: time, evaluation metric value, pipeline run. However, we don't have a unified time series of evaluation results available just yet for the whole pipeline (only on model level). It can be argued that the pipeline performance at time $t$ is the performance of the most up to date model at time $t$. At pipeline level, we do e.g. not care, how an outdated and stale model form the pipeline start performs at the end of the pipeline - it wouldn't be used for inference. Through **post-processing** we can stich the curves of the different models together to get a pipeline performance curve. By doing so we create a unified time series of evaluation results for the whole pipeline run that always uses the newest model. We call this concept `composite model`, as it is a meta model that is composed of the most up to date model at every point in time. For finding the `most up to date model` at every time we use `currently trained` and `currently active` model choice strategies.

In the currently active composite model, every evaluation window/point uses the most recent model that has completed training prior to the start of the evaluation window/point. The currently trained composite model uses the model following the currently active model, potentially benefiting from training on data distributions similar to those in the evaluation set.

> Note: when considering the pipeline as an **ensemble model** of the different models, we can highlight this curve in plot 1 above.

<details>
  <summary><b>Plot 2: (line plot, x=time, y=eval metric (ensemble of all models), color=pipeline)</b></summary>

With that time series per pipeline run, we can already plot the pipelines performance over time for several pipeline runs. We can color code the pipelines to distinguish them.

We expect to see spikes in the pipeline performances when a new model is trained and a stale model is replaced. For pipelines with different trigger policies we expect to see different patterns in the pipeline performance as retraining takes place at different points in time.
</details>

An interesting realization is that when comparing arbitrary different trigger policy pipelines, we cannot rely on equidistant triggering timestamps, as different pipeline runs could have produced evaluations centered around different points in time (currently evaluations are done after training). The Tumbling window approach partially addresses as triggering intervals can be configured independently from the evaluation intervals. However a sliding window approach feels more natural here sampling the pipeline performance at regular intervals of a certain length.

Evaluation at regular intervals would look sth like this:

```mermaid
gantt
    title Pipeline run
    dateFormat YYYY-MM-DD
    axisFormat %Y
    section models
        Trigger1 :milestone, 1930, 0d
        model 1          :a1, 1930, 10y
        Trigger2 :milestone, :after a1
        model 2          :a2, after a1, 5y
        Trigger3 :milestone, :after a2
        model 3          :a3, after a2, 5y
        Trigger4 :milestone, :after a3
        model 4          :a4, after a3, 25y
        Trigger5 :milestone, :after a4
        model 5          :a5, after a4, 25y
        Trigger6 :milestone, :after a5
        model 6          :a6, after a5, 5y
        Trigger7 :milestone, :after a6
        model 7          :a7, after a6, 5y
    section local evaluations (for model 4)
        Evaluation4.1 :milestone, 1950, 0d
        Evaluation4.1 :milestone, 1954, 0d
        Evaluation4.1 :milestone, 1958, 0d
        Evaluation4.1 :milestone, 1962, 0d
        Evaluation4.1 :milestone, 1966, 0d
        Evaluation4.1 :milestone, 1970, 0d
    section local evaluation data (for model 4): +- 5 years
        data4.1 :e4.1, 1947, 6y
        data4.2 :e4.2, 1951, 6y
        data4.3 :e4.3, 1955, 6y
        data4.4 :e4.4, 1959, 6y
        data4.5 :e4.5, 1963, 6y
        data4.6 :e4.6, 1967, 6y
```

<details>
  <summary><b>Plot 4 (matrix plot, x=time up to which samples where used for training, y=model (categorical), color=accuracy of model at this point)</b></summary>

Compared to the current approach this diagram integrates a fair representation of "time" (x-axis). As triggers aren't spaced equally in an abstract trigger policy the current assumptions don't hold anymore. Also this brings the advantage that we would see how drift / model degradation accumulates over **realtime** (not trigger indexes).

</details>

##### 3.2.2.1 Total pipeline performance

Next, we start considering not only the performance of a pipeline, but also the cost of running the pipeline - which is mainly made up of training costs, policy evaluation costs and data loading costs.

Continuing from the previous section, we are left with the following dimensions:

- time
- evaluation metric value
- pipeline run
- cost

For a cost/performance tradeoff comparison there are two useful plotting strategies:

<details>
  <summary><b>Plot 2 + Plot 3 (stacked barplot, x=time, y=cost | eval metric value (ensemble of all models), color=pipeline)</b></summary>

When plotting the cost (cumulative/non-cumulative) over time using the same x-axis scale as the performance plot, we can compare the cost and performance of the different pipelines. We just have to align the plots under each other.

With this plot we can nicely visualize triggering policy decisions at the points in time, where triggers occur,
we can see spikes in training costs. Also, with a cumulative barplot we can compare how
training costs cumulate over time in relation to other cost factors.

</details>

<details>
    <summary><b>Plot 5 (scatterplot, x=total cost (aggregated over time), y=total cost per pipeline run (aggregated over time), color=pipeline)</b></summary>

To unify both dimensions (cost and performance) into one plot, we can plot the total cost of a pipeline run against the total performance of a pipeline run. This allows us to compare the cost/performance tradeoff from different pipelines.

When a user has decided how much he is willing to sacrifice in training time to get rid of a certain amount of performance degradation, this ratio can be visualized in this plot as a straight line that cuts through origin. Policies below this line are not desirable as they overspend. The steeper gradient $y(p) / x(p)$ of a point in the plot, the more efficient the pipeline run is in terms of cost/performance tradeoff. The question remains how we acquire the total (aggregated) performance of a pipeline w.r.t. to a fixed evaluation metric.
</details>

##### 3.2.2.2 Aggregated pipeline performance

As mentioned for plot 5, we need a scalar metric capturing the total performance of a pipeline run. We can define this as a simple average of all `composite model` evaluation values or a time-weighted averaging of the evaluation values from the different evaluation intervals. Rather than using the number of samples as weights (performance should be independent of the actual number of inference queries we get) we use the time the duration where a model was served for inference. This, of course only makes sense when we use evaluation intervals that have non-uniform lengths.

Let's visualize this in the Gantt diagram from 2.1 above:

> E.g. Assume there was no `Trigger5`. Then we should have a really long period where model 4 is used. It becomes clear very fast, that this model has a higher impact on the total aggregated pipeline performance than the a model that was only used for a short period of time (e.g. 1 year).

Using equally sized intervals implicitly solves the problem as we would simply use the model in that long period in multiple evaluation intervals.

For a more explicit way to handle this, we propose to run evaluation request for the intervals between two model trainings capturing the model performance at exactly the time it was served for inference. We can then calculate the pipeline performance as the weighted average of the model performances w.r.t. the served time.

> Note: This time weighted averaging does not make much sense when using a period evaluation strategy as the intervals are equally sized. Then we would only risk misrepresenting the amount of samples periodic evaluation intervals with sliding
windows are allowed to overlap.

#### 3.2.3 Refined pipeline run model

Accounting for the afore mentioned assumptions and the fact that training takes some time, we can visualize a pipeline run as follows:


```mermaid
gantt
    title Pipeline run
    dateFormat YYYY-MM-DD
    axisFormat %Y
    section models
        Trigger1 :milestone, t1, 1930, 0d
        train_batch_1 :a10, 1928, 2y
        training1 :a11, after t1, 1y
        model 1 active          :a12, after a11, until a22

        Trigger2 :milestone, t2, 1940, 0d
        train_batch_2 :a20, after a10, until t2
        training2 :a21, after t2, 1y
        model 2 active          :a22, after a21, until a32

        Trigger3 :milestone, t3, 1945, 0d
        train_batch_3 :a30, after a20, until t3
        training3 :a31, after t3, 1y
        model 3 active          :a32, after a31, until a42

        Trigger4 :milestone, t4, 1950, 0d
        train_batch_4 :a40, after a30, until t4
        training4 :a41, after t4, 1y
        model 4 active          :a42, after a41, 10y
```

The difference to our simplified model earlier is that we not assume `training_time=0` anymore and therefore cannot ignore `training_<i>` . The evaluation intervals would shift by the training times.

> **Remark/Assumptions**: It can be argued though that evaluating one model on the data between two triggers still gives us a good estimate. They are not exactly the same, as we would still use the old model while a training is going on (see diagram below). However, we claim that the difference is negligible as (1) we don't expect much drift between the samples directly after the trigger and after the training (close in time). And (2) compared to the time between two triggers (often years in our case), the duration of a training process is rather small and therefore doesn't have a big impact on the evaluation.

## 3. Summarizing visualization

```mermaid
gantt
    title Pipeline run
    dateFormat YYYY-MM-DD
    axisFormat %Y
    section models
        Trigger1 :milestone, 1930, 0d
        model 1          :a1, 1930, 10y
        Trigger2 :milestone, :after a1
        model 2          :a2, after a1, 5y
        Trigger3 :milestone, :after a2
        model 3          :a3, after a2, 5y
        Trigger4 :milestone, :after a3
        model 4          :a4, after a3, 25y
        Trigger5 :milestone, :after a4
        model 5          :a5, after a4, 25y
        Trigger6 :milestone, :after a5
        model 6          :a6, after a5, 5y
        Trigger7 :milestone, :after a6
        model 7          :a7, after a6, 5y
    section final evaluations
        Evaluation1 :milestone, 1940, 0d
        Evaluation2 :milestone, 1945, 0d
        Evaluation3 :milestone, 1950, 0d
        Evaluation4 :milestone, 1975, 0d
        Evaluation5 :milestone, 2000, 0d
        Evaluation6 :milestone, 2005, 0d
        Evaluation7 :milestone, 2010, 0d
    section final evaluation data
        data1 :e1, 1930, 10y
        data2 :e2, after e1, 5y
        data3 :e3, after e2, 5y
        data4 :e4, after e3, 25y
        data5 :e5, after e4, 25y
        data6 :e6, after e5, 5y
        data7 :e7, after e6, 5y
    section local evaluations (for model 4)
        Evaluation4.1 :milestone, 1950, 0d
        Evaluation4.1 :milestone, 1954, 0d
        Evaluation4.1 :milestone, 1958, 0d
        Evaluation4.1 :milestone, 1962, 0d
        Evaluation4.1 :milestone, 1966, 0d
        Evaluation4.1 :milestone, 1970, 0d
    section local evaluation data (for model 4): +- 5 years
        data4.1 :e4.1, 1947, 6y
        data4.2 :e4.2, 1951, 6y
        data4.3 :e4.3, 1955, 6y
        data4.4 :e4.4, 1959, 6y
        data4.5 :e4.5, 1963, 6y
        data4.6 :e4.6, 1967, 6y
```

## Implementation

### Configuration

To evaluate models you can configure several `EvalHandlerConfig` within the `evaluation` section of the 
pipeline configuration file.

#### Evaluation Strategy

Evaluation strategies describe how the start and end intervals for evaluations are generated.

##### `_IntervalEvalStrategyConfig`

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
