from typing import Annotated, Literal

from pydantic import Field

from modyn.config.schema.pipeline.trigger.drift.metric import BaseMetric


class _EvidentlyPcaDriftMetricMixin(BaseMetric):
    num_pca_component: int | None = Field(None)


class EvidentlyModelDriftMetric(_EvidentlyPcaDriftMetricMixin):
    id: Literal["EvidentlyModelDriftMetric"] = Field("EvidentlyModelDriftMetric")
    bootstrap: bool = Field(False)
    quantile_probability: float = Field(
        0.95, description="Threshold for the evidently api, unused in modyn. Usage discourged."
    )
    threshold: float = Field(0.55, description="Threshold for the evidently api, unused in modyn. Usage discourged.")


class EvidentlyRatioDriftMetric(_EvidentlyPcaDriftMetricMixin):
    id: Literal["EvidentlyRatioDriftMetric"] = Field("EvidentlyRatioDriftMetric")
    component_stattest: str = Field(
        "wasserstein",
        description="The statistical test used to compare the components.",
    )
    component_stattest_threshold: float = Field(
        0.1, description="Threshold for the evidently api, unused in modyn. Usage discourged."
    )
    threshold: float = Field(0.2, description="Threshold for the evidently api, unused in modyn. Usage discourged.")


class EvidentlySimpleDistanceDriftMetric(_EvidentlyPcaDriftMetricMixin):
    id: Literal["EvidentlySimpleDistanceDriftMetric"] = Field("EvidentlySimpleDistanceDriftMetric")
    distance_metric: str = Field(
        "euclidean",
        description="The distance metric used for the distance calculation.",
    )
    bootstrap: bool = Field(False)
    quantile_probability: float = Field(
        0.95, description="Threshold for the evidently api, unused in modyn. Usage discourged."
    )
    threshold: float = Field(0.2, description="Threshold for the evidently api, unused in modyn. Usage discourged.")


class EvidentlyHellingerDistanceDriftMetric(BaseMetric):
    id: Literal["EvidentlyHellingerDistanceDriftMetric"] = Field("EvidentlyHellingerDistanceDriftMetric")


EvidentlyDriftMetric = Annotated[
    EvidentlyModelDriftMetric
    | EvidentlyRatioDriftMetric
    | EvidentlySimpleDistanceDriftMetric
    | EvidentlyHellingerDistanceDriftMetric,
    Field(discriminator="id"),
]
