from typing import Optional


class ModelStorageStrategyConfig:
    """
    Helper class to represent the configuration of a model storage strategy.
    """

    def __init__(self, name: str):
        self.name = name
        self.zip: Optional[bool] = None
        self.zip_algorithm: Optional[str] = None
        self.config: Optional[str] = None
