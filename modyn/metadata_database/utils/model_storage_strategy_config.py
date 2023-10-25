from dataclasses import dataclass
from typing import Optional

# pylint: disable=no-name-in-module
from modyn.selector.internal.grpc.generated.selector_pb2 import StrategyConfig


@dataclass
class ModelStorageStrategyConfig:
    """
    This class holds all information of a generic model storage strategy.
    It is used to insert a given strategy in the metadata database.
    """

    name: str
    zip: bool = False
    zip_algorithm: Optional[str] = None
    config: Optional[str] = None

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def from_config(cls, strategy_config: StrategyConfig):  # type: ignore[no-untyped-def]
        strategy = cls(strategy_config.name)
        if strategy_config.HasField("zip") and strategy_config.zip is not None:
            strategy.zip = strategy_config.zip
        if strategy_config.HasField("zip_algorithm") and strategy_config.zip is not None:
            strategy.zip_algorithm = strategy_config.zip_algorithm
        if strategy_config.HasField("config") and strategy_config.config is not None:
            strategy.config = strategy_config.config.value
        return strategy
