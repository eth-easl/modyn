from typing import Annotated, Literal, Union

from modyn.config.schema.base_model import ModynBaseModel
from pydantic import Field


class _EvidentlyBaseDriftMetric(ModynBaseModel):
    num_pca_component: int | None = Field(None)


class EvidentlyMmdDriftMetric(_EvidentlyBaseDriftMetric):
    id: Literal["EvidentlyMmdDriftMetric"] = Field("EvidentlyMmdDriftMetric")
    bootstrap: bool = Field(False)
    quantile_probability: float = 0.05
    threshold: float = Field(0.15)


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
        EvidentlyMmdDriftMetric,
        EvidentlyModelDriftMetric,
        EvidentlyRatioDriftMetric,
        EvidentlySimpleDistanceDriftMetric,
    ],
    Field(discriminator="id"),
]
