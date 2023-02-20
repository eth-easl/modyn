from typing import Any, Sequence
from torch import nn
import numpy as np

from modyn.models.dlrm.nn.factories import create_interaction
from modyn.models.dlrm.nn.parts import DlrmBottom, DlrmTop
from modyn.models.dlrm.utils.feature_spec import FeatureSpec, get_device_mapping, get_embedding_sizes

class DLRM(nn.Module):
    def __init__(
        self,
        model_configuration: dict[str, Any],
    ) -> None:

        super().__init__()



        feature_spec = FeatureSpec.from_yaml("feature_spec.yaml")

        world_embedding_sizes = get_embedding_sizes(feature_spec, max_table_size=model_configuration["max_table_size"])
        world_categorical_feature_sizes = np.asarray(world_embedding_sizes)
        device_mapping = get_device_mapping(world_embedding_sizes, num_gpus=1)

        # Embedding sizes for each GPU
        categorical_feature_sizes = world_categorical_feature_sizes[device_mapping['embedding'][0]].tolist()
        num_numerical_features = feature_spec.get_number_of_numerical_features()

        self._vectors_per_gpu = device_mapping['vectors_per_gpu']
        self._embedding_device_mapping = device_mapping['embedding']
        self._embedding_dim = model_configuration['embedding_dim']
        self._interaction_op = model_configuration['interaction_op']
        self._hash_indices = model_configuration['hash_indices']

        # TODO(fotstrt): fix this
        interaction = create_interaction(self._interaction_op, len(world_categorical_feature_sizes)
, self._embedding_dim)

        # ignore device here since it is handled by the trainer
        self.bottom_model = DlrmBottom(
            num_numerical_features,
            categorical_feature_sizes,
            model_configuration['bottom_mlp_sizes'],
            model_configuration['embedding_type'],
            self._embedding_dim,
            hash_indices=self._hash_indices,
            use_cpp_mlp=model_configuration['use_cpp_mlp'],
            fp16=model_configuration['fp16']
        )

        self.top_model = DlrmTop(
            model_configuration['top_mlp_sizes'],
            interaction,
            use_cpp_mlp=model_configuration['use_cpp_mlp']
        )


    def extra_repr(self):
        return f"interaction_op={self._interaction_op}, hash_indices={self._hash_indices}"

    @classmethod
    def from_dict(cls, obj_dict, **kwargs):
        """Create from json str"""
        return cls(**obj_dict, **kwargs)


    def forward(self, input):
        """
        Args:
            input: a dict containing:
                numerical_input (Tensor): with shape [batch_size, num_numerical_features]
                categorical_inputs (Tensor): with shape [batch_size, num_categorical_features]
            batch_sizes_per_gpu (Sequence[int]):
        """
        numerical_input = input['numerical_input']
        categorical_inputs = input['categorical_inputs']

        # bottom mlp output may be not present before all to all communication
        from_bottom, bottom_mlp_output = self.bottom_model(numerical_input, categorical_inputs)
        return self.top_model(from_bottom, bottom_mlp_output)
