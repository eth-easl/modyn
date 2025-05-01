from typing import Any

import torch
from megatron import get_args, initialize_megatron

# Megatron imports (assuming Megatron-LM is installed / in PYTHONPATH)
from megatron.arguments import parse_args
from megatron.model import GPTModel
from megatron.training import setup_model_and_optimizer
from torch import nn

from modyn.models.coreset_methods_support import CoresetSupportingModule


class MegatronGPT:
    # pylint: disable-next=unused-argument
    def __init__(self, hparams: Any, device: str, amp: bool) -> None:
        """
        hparams is expected to have attributes (or keys) that define the usual
        Megatron arguments, e.g.:
          - num_layers, hidden_size, num_attention_heads
          - seq_length, max_position_embeddings
          - micro_batch_size, global_batch_size, train_iters
          - lr, min_lr, lr_decay_style
          - vocab_file, merge_file
          - data_path, data_impl, task, etc.
        If something is missing, we use a GPT-2–ish default.
        """
        self.model = MegatronGPTModyn(hparams)
        self.model.to(device)

    def get_model(self):
        """Return the underlying Megatron model if needed."""
        return self.model


class MegatronGPTModyn(CoresetSupportingModule):
    """
    Analogous to 'Gpt2Modyn': the actual module that wraps Megatron's GPTModel.
    """

    def __init__(self, hparams: Any) -> None:
        """
        1) Convert user-supplied hparams to Megatron command-line arguments.
        2) Initialize Megatron, parse args.
        3) Create a GPTModel, store it in self.model.
        """
        super().__init__()

        # 1) Build command-line arguments from hparams
        cmd_args = self._build_megatron_args(hparams)

        # 2) Parse & initialize Megatron (global)
        args = parse_args(cmd_args)
        initialize_megatron(args)

        # 3) Build the GPT model & optimizer
        #    In single-GPU mode (tensor & pipeline parallel size = 1),
        #    it returns lists of length 1 for model, optimizer, etc.
        model, optimizer, lr_scheduler = setup_model_and_optimizer(self._model_provider)
        self.megatron_model = model[0]
        self.megatron_optimizer = optimizer[0]
        self.megatron_lr_scheduler = lr_scheduler
        self.megatron_model.train()

        # In your original GPT-2 code: self.model = GPT2LMHeadModel(...)
        # Here we store the actual Megatron GPT model in self.model for consistency:
        self.model = self.megatron_model

    @staticmethod
    def _model_provider():
        """
        This function is called by Megatron to construct the GPT model
        with hyperparams from `get_args()`.
        """

        args = get_args()
        # GPTModel constructor in Megatron typically uses
        #   num_layers = args.num_layers,
        #   hidden_size = args.hidden_size,
        #   num_attention_heads = args.num_attention_heads, etc.
        # Because these are read from the global 'args'.
        return GPTModel(
            num_tokentypes=0,
            parallel_output=True,  # For model parallel, still fine on single GPU
            pre_process=True,
            post_process=True,
        )

    @staticmethod
    def _build_megatron_args(hparams: Any) -> list[str]:
        """
        Convert Python `hparams` into a list of strings that mimic
        Megatron command-line arguments.
        Provide GPT-2–like defaults if certain fields are missing.
        """
        defaults = {
            "num_layers": 12,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "seq_length": 1024,
            "max_position_embeddings": 1024,
            "micro_batch_size": 2,
            "global_batch_size": 2,
            "train_iters": 1000,
            "lr": 1e-4,
            "min_lr": 1e-5,
            "lr_decay_style": "linear",
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "vocab_file": "./vocab.json",
            "merge_file": "./merges.txt",
            "data_impl": "infer",
            "data_path": "./my_data",  # Typically a prefix or path to training data
            "task": "GPT",
            "fp16": True,
        }

        # Merge user hparams over defaults
        # If hparams is an object with attributes, we can do: getattr(hparams, "field", defaults[field])
        # If it's a dict, we do hparams.get(...)
        for key, val in defaults.items():
            if hasattr(hparams, key):
                defaults[key] = getattr(hparams, key)
            else:
                defaults[key] = hparams.get(key, val)

        # Build the command list
        cmd_args = [
            "--num-layers",
            str(defaults["num_layers"]),
            "--hidden-size",
            str(defaults["hidden_size"]),
            "--num-attention-heads",
            str(defaults["num_attention_heads"]),
            "--seq-length",
            str(defaults["seq_length"]),
            "--max-position-embeddings",
            str(defaults["max_position_embeddings"]),
            "--micro-batch-size",
            str(defaults["micro_batch_size"]),
            "--global-batch-size",
            str(defaults["global_batch_size"]),
            "--train-iters",
            str(defaults["train_iters"]),
            "--tensor-model-parallel-size",
            str(defaults["tensor_model_parallel_size"]),
            "--pipeline-model-parallel-size",
            str(defaults["pipeline_model_parallel_size"]),
            "--lr",
            str(defaults["lr"]),
            "--min-lr",
            str(defaults["min_lr"]),
            "--lr-decay-style",
            str(defaults["lr_decay_style"]),
            "--vocab-file",
            defaults["vocab_file"],
            "--merge-file",
            defaults["merge_file"],
            "--data-impl",
            defaults["data_impl"],
            "--data-path",
            defaults["data_path"],
            "--task",
            defaults["task"],
        ]
        if defaults["fp16"]:
            cmd_args.append("--fp16")

        return cmd_args

    def forward(self, data: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward method for text generation or language modeling tasks.

        Args:
            data (torch.Tensor): shape (batch_size, seq_len, 2),
                where the last dimension has token IDs and attention masks.
            labels (torch.Tensor, optional): not directly used in Megatron’s GPTModel forward
                (Megatron typically returns logits; you do the CE loss externally).
        Returns:
            The output logits from the GPT model: shape (batch_size, seq_len, vocab_size)
        """
        input_ids = data[:, :, 0]
        attention_mask = data[:, :, 1]

        # Megatron’s GPTModel forward typically wants:
        #   forward(tokens, position_ids=None, attention_mask=None, ...)
        logits = self.megatron_model(input_ids, attention_mask=attention_mask, labels=labels)
        return logits

    def get_last_layer(self) -> nn.Module:
        """
        Retrieve the final "lm_head" equivalent in Megatron GPT.
        Megatron’s GPTModel typically merges the final linear projection into model.output_layer.
        """
        return self.megatron_model.output_layer
