import time

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modyn.config.schema.pipeline import LLMScoreMetricConfig
from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class LLMScore(AbstractHolisticMetric):
    """LLM-based evaluation metric for response correctness, coherence, and factuality."""

    def __init__(self, config: LLMScoreMetricConfig) -> None:
        super().__init__(config)
        self.evaluation_result: float | None = None
        self.use_api: bool = config.get("use_api", False)  # Whether to use API or local model
        self.model_name: str = config.get("model", "meta-llama/Llama-3.3-70B-Instruct")
        self.api_url: str | None = config.get("api_url", None) if self.use_api else None

        # Default prompt now compares ground truth and prediction.
        self.evaluation_prompt: str = config.get(
            "evaluation_prompt",
            "Evaluate how close each predicted response is to its ground truth. Provide a numerical score between 0 and 1 for each pair. Return only the numerical scores on separate lines.",
        )

        # Set batch size with a default value of 10.
        self.batch_size: int = config.get("batch_size", 10)

        if not self.use_api:
            self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

    def _dataset_evaluated_callback(self, y_true: list[str], y_pred: list[str], num_samples: int) -> None:
        """Evaluate responses using an LLM and assign a correctness score."""
        assert self.evaluation_result is None

        if self.use_api:
            scores = self._evaluate_via_api(y_true, y_pred)
        else:
            scores = self._evaluate_locally(y_true, y_pred)
        self.evaluation_result = sum(scores) / num_samples

    def _evaluate_via_api(self, y_true: list[str], y_pred: list[str]) -> list[float]:
        """Use API to evaluate batches of responses by comparing ground truth and predictions."""
        if not self.api_url:
            raise ValueError("API URL must be provided for API-based evaluation.")
        scores = []
        for i in range(0, len(y_true), self.batch_size):
            batch_true = y_true[i : i + self.batch_size]
            batch_pred = y_pred[i : i + self.batch_size]
            eval_prompt = self.evaluation_prompt + "\n\n"
            for j, (gt, pred) in enumerate(zip(batch_true, batch_pred)):
                eval_prompt += f"{j+1}. Ground Truth: {gt}\n   Predicted: {pred}\n"
            while True:
                try:
                    # pylint: disable=missing-timeout
                    response = requests.post(
                        self.api_url,
                        json={"model": self.model_name, "messages": [{"content": eval_prompt, "role": "user"}]},
                    )
                    response.raise_for_status()
                    break  # Exit loop if no error is encountered.
                except Exception:  # pylint: disable=broad-exception-caught
                    time.sleep(5)  # Wait 5 seconds before retrying on error.
            results = response.json()["choices"][0]["message"]["content"].strip().split("\n")
            for res in results:
                try:
                    scores.append(float(res.strip()))
                except ValueError:
                    scores.append(0.0)  # Handle invalid output
        return scores

    def _evaluate_locally(self, y_true: list[str], y_pred: list[str]) -> list[float]:
        """Use a local Llama 3.3 model on an H100 GPU for evaluation in batches."""
        scores = []
        for i in range(0, len(y_true), self.batch_size):
            batch_true = y_true[i : i + self.batch_size]
            batch_pred = y_pred[i : i + self.batch_size]
            prompts = [
                f"Evaluate how close the predicted response is to its ground truth:\n\nGround Truth: {gt}\nPredicted: {pred}\nScore:"
                for gt, pred in zip(batch_true, batch_pred)
            ]
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_length=20)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                try:
                    scores.append(float(d.strip()))
                except ValueError:
                    scores.append(0.0)  # Handle invalid output
        return scores

    def get_evaluation_result(self) -> float:
        assert self.evaluation_result is not None
        return self.evaluation_result

    def get_name(self) -> str:
        return "LLM-Score"
