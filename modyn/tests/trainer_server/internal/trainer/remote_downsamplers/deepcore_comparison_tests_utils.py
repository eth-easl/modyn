import numpy as np
import torch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from torch import nn


class DummyModel(CoresetSupportingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer = nn.Linear(in_features=1, out_features=10)
        self.output_layer = nn.Linear(in_features=10, out_features=1)

    def forward(self, input_tensor):
        input_tensor = torch.relu(self.hidden_layer(input_tensor))
        input_tensor = self.embedding_recorder(input_tensor)
        outputs = self.output_layer(input_tensor)
        return outputs

    def get_last_layer(self):
        return self.output_layer


def assert_close_matrices(matrix1, matrix2):
    for row1, row2 in zip(matrix1, matrix2):
        assert len(row1) == len(row2)
        for el1, el2 in zip(row1, row2):
            assert np.isclose(el1, el2, 1e-2)
