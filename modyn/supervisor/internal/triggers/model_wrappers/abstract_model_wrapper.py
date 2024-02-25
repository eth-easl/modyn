from abc import abstractmethod
import pandas as pd
import torch

import logging

logger = logging.getLogger(__name__)


class AbstractModelWrapper:
    def __init__(self):
        pass
            
    @abstractmethod
    def get_embeddings(self, dataloader) -> torch.Tensor:
        """computes the embeddings."""
        raise NotImplementedError

    @abstractmethod
    def get_embeddings_evidently_format(self, dataloader) -> pd.DataFrame:
        """computes the embeddings and convert to evidently format."""
        raise NotImplementedError

