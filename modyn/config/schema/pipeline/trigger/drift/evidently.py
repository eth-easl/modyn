from typing import Annotated, Literal, Union

from pydantic import Field

from modyn.config.schema.pipeline.trigger.drift.metric import BaseMetric


class _EvidentlyBaseDriftMetric(BaseMetric):
    num_pca_component: int | None = Field(None)


class EvidentlyModelDriftMetric(_EvidentlyBaseDriftMetric):
    id: Literal["EvidentlyModelDriftMetric"] = Field("EvidentlyModelDriftMetric")
    bootstrap: bool = Field(False)
    quantile_probability: float = 0.95
    threshold: float = Field(0.55)


class EvidentlyRatioDriftMetric(_EvidentlyBaseDriftMetric):
    id: Literal["EvidentlyRatioDriftMetric"] = Field("EvidentlyRatioDriftMetric")
    component_stattest: str = Field("wasserstein", description="The statistical test used to compare the components.")
    component_stattest_threshold: float = Field(0.1)
    threshold: float = Field(0.2)


class EvidentlySimpleDistanceDriftMetric(_EvidentlyBaseDriftMetric):
    id: Literal["EvidentlySimpleDistanceDriftMetric"] = Field("EvidentlySimpleDistanceDriftMetric")
    distance_metric: str = Field("euclidean", description="The distance metric used for the distance calculation.")
    bootstrap: bool = Field(False)
    quantile_probability: float = 0.95
    threshold: float = Field(0.2)


EvidentlyDriftMetric = Annotated[
    Union[
        EvidentlyModelDriftMetric,
        EvidentlyRatioDriftMetric,
        EvidentlySimpleDistanceDriftMetric,
    ],
    Field(discriminator="id"),
]
