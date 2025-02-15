import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modyn.config.schema.pipeline import LLMScoreMetricConfig
from modyn.evaluator.internal.metrics.abstract_holistic_metric import AbstractHolisticMetric


class LLMScore(AbstractHolisticMetric):
    """LLM-based evaluation metric for response correctness, coherence, and factuality."""

    def __init__(self, config: LLMScoreMetricConfig) -> None:
        super().__init__(config)
        self.evaluation_result: float | None = None
        self.use_api = config.get("use_api", False)  # Whether to use OpenAI API or local model
        self.model_name = config.get("model", "meta-llama/Llama-2-13b-chat-hf")
        self.api_key = config.get("api_key", None) if self.use_api else None

        if not self.use_api:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

    def _dataset_evaluated_callback(self, prompts: list[str], responses: list[str], num_samples: int) -> None:
        """Evaluate responses using an LLM and assign a correctness score."""
        assert self.evaluation_result is None

        if self.use_api:
            self.evaluation_result = self._evaluate_via_api(prompts, responses)
        else:
            self.evaluation_result = self._evaluate_locally(prompts, responses)

    def _evaluate_via_api(self, prompts, responses):
        """Call OpenAI's GPT-4 Turbo API for evaluation."""
        import openai

        openai.api_key = self.api_key

        evaluation_prompt = """
        You are an AI judge. Given user prompts and AI-generated responses, evaluate each response on:
        - Correctness (does it answer the question?)
        - Coherence (is it well-structured?)
        - Factuality (is it factually accurate?)

        Assign a score (1-10) to each response."""

        # Prompts and Responses:
        # {''.join([f"Prompt {i+1}: {p}\nResponse {i+1}: {r}\n" for i, (p, r) in enumerate(zip(prompts, responses))])}

        # Provide JSON output: [{{"response_id": 1, "score": 8}}, ...]

        completion = openai.ChatCompletion.create(
            model="gpt-4-turbo", messages=[{"role": "user", "content": evaluation_prompt}]
        )
        return json.loads(completion["choices"][0]["message"]["content"])

    def _evaluate_locally(self, prompts, responses):
        """Use a local Llama 2 model on an H100 GPU for evaluation."""
        scores = []
        for prompt, response in zip(prompts, responses):
            input_text = f"Evaluate the following response on correctness, coherence, and factuality:\n\nPrompt: {prompt}\nResponse: {response}\nScore:"
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=20)

            score = self.tokenizer.decode(output[0], skip_special_tokens=True)
            try:
                scores.append(float(score.strip()))
            except ValueError:
                scores.append(0)  # Handle invalid output

        return sum(scores) / len(scores)

    def get_evaluation_result(self) -> float:
        assert self.evaluation_result is not None
        return self.evaluation_result

    def get_name(self) -> str:
        return "LLM-Score"
