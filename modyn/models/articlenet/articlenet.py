# pylint: disable=W0223
from typing import Any

import torch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import nn
from transformers import DistilBertModel


class ArticleNet:
    """
    Adapted from WildTime. This network is used for NLP tasks (Arxiv and Huffpost)
    Here you can find the original implementation:
    https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/networks/article.py
    """

    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = ArticleNetwork(**model_configuration)
        self.model.to(device)


class DistilBertFeaturizer(DistilBertModel):  # pylint: disable=abstract-method
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # slice the input tensor to get input ids and attention mask
        # The model receives as input the output of the tokenizer, where the first dimension
        # contains the tokens and the second a boolean mask to indicate which tokens are valid
        # (the sentences have different lengths but the output of the tokenizer has always the same size,
        # so you need the mask to understand what is useful data and what is just padding)
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        # DistilBert's forward pass
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[
            0
        ]  # 0: last hidden state, 1: hiddent states, 2: attentions
        pooled_output = hidden_state[:, 0]  # first token is the pooled output, which is the aggregated representation
        # of the entire input sequence
        return pooled_output

    def _reorder_cache(self, past: Any, beam_idx: Any) -> None:
        pass


class ArticleNetwork(CoresetSupportingModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.featurizer.d_out, num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        embedding = self.featurizer(data)
        embedding = self.embedding_recorder(embedding)
        return self.classifier(embedding)

    def get_last_layer(self) -> nn.Module:
        return self.classifier
