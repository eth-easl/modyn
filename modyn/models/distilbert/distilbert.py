from typing import Any

import torch
from torch import nn
from transformers import DistilBertForSequenceClassification, DistilBertModel


class DistilBertNet:
    # pylint: disable-next=unused-argument
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = ArticleNetwork(**model_configuration)
        self.model.to(device)


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


class ArticleNetwork(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        classifier = nn.Linear(featurizer.d_out, num_classes)
        self.model = nn.Sequential(featurizer, classifier)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)
