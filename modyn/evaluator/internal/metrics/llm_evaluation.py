import time
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modyn.config.schema.pipeline import RougeMetricConfig
from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class LLMEvaluation(AbstractHolisticMetric):
    """
    LLM-based evaluation metric with two evaluation modes:
      - "boolean": returns 'true' or 'false' per sample (final score is the average where true=1 and false=0)
      - "numeric": returns a numerical score (0 to 100) per sample
    Default values and prompts are set similar to the presampling strategy.
    """

    def __init__(self, config: RougeMetricConfig) -> None:
        super().__init__(config)
        self.evaluation_result: float | None = None
        self.use_api: bool = config.get("use_api", False)
        self.model_name: str = config.get("model", "meta-llama/Llama-3.3-70B-Instruct")
        self.api_url: Any | None = config.get("api_url", None) if self.use_api else None
        self.evaluation_mode: str = config.get("evaluation_mode", "boolean")  # "boolean" or "numeric"
        self.batch_size: int = config.get("batch_size", 10)

        # Set default prompt based on evaluation mode.
        if self.evaluation_mode == "numeric":
            self.evaluation_prompt: str = config.get(  # type: ignore
                "evaluation_prompt",
                "Evaluate how close each predicted response is to its ground truth. "
                "Provide a numerical score between 0 and 100 for each pair. "
                "Return only the numerical scores on separate lines.",
            )
        else:  # boolean mode
            self.evaluation_prompt: str = config.get(  # type: ignore
                "evaluation_prompt",
                "Determine if the following texts are useful for training an LLM on Wikipedia fact updates.\n\n",
            )

        if not self.use_api:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

    def _dataset_evaluated_callback(self, y_true: list[str], y_pred: list[str], num_samples: int) -> None:
        """Evaluate responses using an LLM and assign a final evaluation result."""
        assert self.evaluation_result is None

        if self.use_api:
            scores = self._evaluate_via_api(y_true, y_pred)
        else:
            scores = self._evaluate_locally(y_true, y_pred)

        if self.evaluation_mode == "boolean":
            # Convert string results: 'true' -> 1.0; anything else -> 0.0
            numeric_scores = [1.0 if str(s).strip().lower() == "true" else 0.0 for s in scores]
        else:
            numeric_scores = []
            for s in scores:
                try:
                    numeric_scores.append(float(s))
                except Exception:
                    numeric_scores.append(0.0)
        self.evaluation_result = sum(numeric_scores) / num_samples

    def _evaluate_via_api(self, y_true: list[str], y_pred: list[str]) -> list[str]:
        scores = []
        for i in range(0, len(y_true), self.batch_size):
            batch_true = y_true[i : i + self.batch_size]
            batch_pred = y_pred[i : i + self.batch_size]
            if self.evaluation_mode == "boolean":
                prompt = self.evaluation_prompt + "\n"
                for j, (gt, pred) in enumerate(zip(batch_true, batch_pred)):
                    prompt += f"{j+1}. Ground Truth: {gt}\n   Predicted: {pred}\n"
                prompt += "\nReturn only 'true' or 'false' responses for each pair, one per line."
            else:
                prompt = self.evaluation_prompt + "\n"
                for j, (gt, pred) in enumerate(zip(batch_true, batch_pred)):
                    prompt += f"{j+1}. Ground Truth: {gt}\n   Predicted: {pred}\n"
                prompt += "\nReturn only the numerical scores (0 to 100) on separate lines."
            while True:
                try:
                    response = requests.post(
                        self.api_url,  # type: ignore
                        json={"model": self.model_name, "messages": [{"content": prompt, "role": "user"}]},
                    )
                    response.raise_for_status()
                    break
                except Exception:
                    time.sleep(5)
            results = response.json()["choices"][0]["message"]["content"].strip().split("\n")
            for res in results:
                scores.append(res.strip())
        return scores

    def _evaluate_locally(self, y_true: list[str], y_pred: list[str]) -> list[str]:
        scores = []
        for i in range(0, len(y_true), self.batch_size):
            batch_true = y_true[i : i + self.batch_size]
            batch_pred = y_pred[i : i + self.batch_size]
            if self.evaluation_mode == "boolean":
                prompts = [
                    f"Determine if the following pair is useful:\n\nGround Truth: {gt}\nPredicted: {pred}\n\nReturn only 'true' or 'false'."
                    for gt, pred in zip(batch_true, batch_pred)
                ]
            else:
                prompts = [
                    f"Evaluate how close the predicted response is to its ground truth:\n\nGround Truth: {gt}\nPredicted: {pred}\n\nReturn a numerical score between 0 and 100."
                    for gt, pred in zip(batch_true, batch_pred)
                ]
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_length=20)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                scores.append(d.strip())
        return scores

    def get_evaluation_result(self) -> float:
        assert self.evaluation_result is not None
        return self.evaluation_result

    def get_name(self) -> str:
        return "LLM-Evaluation"
