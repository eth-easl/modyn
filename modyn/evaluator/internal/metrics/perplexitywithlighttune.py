import numpy as np
import torch

from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class PerplexityWithLightTuning(AbstractHolisticMetric):
    """Perplexity metric that includes light-tuning before evaluation."""

    def __init__(self, config: PerplexityMetricConfig) -> None:
        super().__init__(config)
        self.total_loss = 0.0
        self.total_tokens = 0
        self.model = None  # Placeholder for the model

    def light_tune_model(self, model, tuning_data):
        """Perform light-tuning on the model before evaluating perplexity."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()
        for batch in tuning_data:
            optimizer.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        self.model = model

    def _dataset_evaluated_callback(self, y_true: torch.Tensor, y_pred: torch.Tensor, num_samples: int) -> None:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fn(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
        self.total_loss += loss.item()
        self.total_tokens += y_true.numel()

    def get_evaluation_result(self) -> float:
        if self.total_tokens == 0:
            self.warning("Did not see any samples.")
            return float("inf")
        return np.exp(self.total_loss / self.total_tokens)

    def get_name(self) -> str:
        return "Perplexity with Light-Tuning"
