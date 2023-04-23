import subprocess
from pathlib import Path
from typing import Any

import modyn
import torch
from torch import nn


class RNN:
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = RNNModel(model_configuration, device, amp)
        self.model.to(device)


class RNNModel(nn.Module):
    """RNN.

    This RNN is to classify text data.
    In particular, it is used for the Reddit benchmark.
    """

    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        super(RNNModel, self).__init__()

        self.validate_config(model_configuration, device)
        self.n_layers = model_configuration["n_layers"]
        self.hidden_dim = model_configuration["hidden_dim"]

        if model_configuration["interaction_op"] == "cuda_dot":
            install_cuda_extensions_if_not_present()

        self.embedding_layer = nn.Embedding(
            num_embeddings=len(model_configuration["vocab"]), embedding_dim=model_configuration["embedding_len"]
        )
        self.rnn = nn.RNN(
            input_size=model_configuration["embedding_len"],
            hidden_size=model_configuration["hidden_dim"],
            num_layers=model_configuration["n_layers"],
            batch_first=True,
            nonlinearity="relu",
            dropout=0.2,
        )
        self.linear = nn.Linear(model_configuration["hidden_dim"], len(model_configuration["target_classes"]))

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(embeddings, torch.randn(self.n_layers, len(X_batch), self.hidden_dim))
        return self.linear(output[:, -1])


# FIXME later merge with dlrm
def install_cuda_extensions_if_not_present() -> None:
    modyn_base_path = Path(modyn.__path__[0])
    rnn_path = modyn_base_path / "models" / "rnn"
    rnn_cuda_ext_path = rnn_path / "cuda_ext"
    shared_libraries = sorted(list(rnn_cuda_ext_path.glob("*.so")))
    shared_libraries_names = [lib.name.split(".")[0] for lib in shared_libraries]

    if shared_libraries_names != ["fused_embedding", "interaction_ampere", "interaction_volta", "sparse_gather"]:
        # install
        subprocess.run(["pip", "install", "-v", "-e", "."], check=True, cwd=rnn_path)
