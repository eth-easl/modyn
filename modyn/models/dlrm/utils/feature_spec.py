# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# flake8: noqa
# mypy: ignore-errors

import math
import os
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml
from modyn.models.dlrm.utils.data_defaults import (
    CARDINALITY_SELECTOR,
    CATEGORICAL_CHANNEL,
    DTYPE_SELECTOR,
    FEATURES_SELECTOR,
    FILES_SELECTOR,
    LABEL_CHANNEL,
    NUMERICAL_CHANNEL,
    SPLIT_BINARY,
    TEST_MAPPING,
    TRAIN_MAPPING,
    TYPE_SELECTOR,
    get_categorical_feature_type,
)

""" For performance reasons, numerical features are required to appear in the same order
    in both source_spec and channel_spec.
    For more detailed requirements, see the check_feature_spec method"""


class FeatureSpec:
    def __init__(self, feature_spec=None, source_spec=None, channel_spec=None, metadata=None, base_directory=None):
        self.feature_spec: Dict = feature_spec if feature_spec is not None else {}
        self.source_spec: Dict = source_spec if source_spec is not None else {}
        self.channel_spec: Dict = channel_spec if channel_spec is not None else {}
        self.metadata: Dict = metadata if metadata is not None else {}
        self.base_directory: str = base_directory

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as feature_spec_file:
            base_directory = os.path.dirname(path)
            feature_spec = yaml.safe_load(feature_spec_file)
            return cls.from_dict(feature_spec, base_directory=base_directory)

    @classmethod
    def from_dict(cls, source_dict, base_directory):
        return cls(base_directory=base_directory, **source_dict)

    def to_dict(self) -> Dict:
        attributes_to_dump = ["feature_spec", "source_spec", "channel_spec", "metadata"]
        return {attr: self.__dict__[attr] for attr in attributes_to_dump}

    def to_string(self):
        return yaml.dump(self.to_dict())

    def to_yaml(self, output_path=None):
        if not output_path:
            output_path = self.base_directory + "/feature_spec.yaml"
        with open(output_path, "w") as output_file:
            print(yaml.dump(self.to_dict()), file=output_file)

    def get_number_of_numerical_features(self) -> int:
        numerical_features = self.channel_spec[NUMERICAL_CHANNEL]
        return len(numerical_features)

    def cat_positions_to_names(self, positions: List[int]):
        #  Ordering needs to correspond to the one in get_categorical_sizes()
        feature_names = self.get_categorical_feature_names()
        return [feature_names[i] for i in positions]

    def get_categorical_feature_names(self):
        """Provides the categorical feature names. The returned order should me maintained."""
        return self.channel_spec[CATEGORICAL_CHANNEL]

    def get_categorical_sizes(self) -> List[int]:
        """For a given feature spec, this function is expected to return the sizes in the order corresponding to the
        order in the channel_spec section"""
        categorical_features = self.get_categorical_feature_names()
        cardinalities = [self.feature_spec[feature_name][CARDINALITY_SELECTOR] for feature_name in categorical_features]

        return cardinalities

    def check_feature_spec(self):
        # TODO check if cardinality fits in dtype, check if base directory is set
        # TODO split into two checking general and model specific requirements
        # check that mappings are the ones expected
        mapping_name_list = list(self.source_spec.keys())
        assert sorted(mapping_name_list) == sorted([TEST_MAPPING, TRAIN_MAPPING])

        # check that channels are the ones expected
        channel_name_list = list(self.channel_spec.keys())
        assert sorted(channel_name_list) == sorted([CATEGORICAL_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL])

        categorical_features_list = self.channel_spec[CATEGORICAL_CHANNEL]
        numerical_features_list = self.channel_spec[NUMERICAL_CHANNEL]
        label_features_list = self.channel_spec[LABEL_CHANNEL]
        set_of_categorical_features = set(categorical_features_list)
        set_of_numerical_features = set(numerical_features_list)

        # check that exactly one label feature is selected
        assert len(label_features_list) == 1
        label_feature_name = label_features_list[0]

        # check that lists in channel spec contain unique names
        assert sorted(list(set_of_categorical_features)) == sorted(categorical_features_list)
        assert sorted(list(set_of_numerical_features)) == sorted(numerical_features_list)

        # check that all features used in channel spec are exactly ones defined in feature_spec
        feature_spec_features = list(self.feature_spec.keys())
        channel_spec_features = list(
            set.union(set_of_categorical_features, set_of_numerical_features, {label_feature_name})
        )
        assert sorted(feature_spec_features) == sorted(channel_spec_features)

        # check that correct dtypes are provided for all features
        for feature_dict in self.feature_spec.values():
            assert DTYPE_SELECTOR in feature_dict
            try:
                np.dtype(feature_dict[DTYPE_SELECTOR])
            except TypeError:
                assert False, "Type not understood by numpy"

        # check that categorical features have cardinality provided
        for feature_name, feature_dict in self.feature_spec.items():
            if feature_name in set_of_categorical_features:
                assert CARDINALITY_SELECTOR in feature_dict
                assert isinstance(feature_dict[CARDINALITY_SELECTOR], int)

        for mapping_name in [TRAIN_MAPPING, TEST_MAPPING]:
            mapping = self.source_spec[mapping_name]
            mapping_features = set()
            for chunk in mapping:
                # check that chunk has the correct type
                assert chunk[TYPE_SELECTOR] == SPLIT_BINARY

                contained_features = chunk[FEATURES_SELECTOR]
                containing_files = chunk[FILES_SELECTOR]

                # check that features are unique in mapping
                for feature in contained_features:
                    assert feature not in mapping_features
                    mapping_features.add(feature)

                # check that chunk has at least one features
                assert len(contained_features) >= 1

                # check that chunk has exactly file
                assert len(containing_files) == 1

                first_feature = contained_features[0]

                if first_feature in set_of_categorical_features:
                    # check that each categorical feature is in a different file
                    assert len(contained_features) == 1

                elif first_feature in set_of_numerical_features:
                    # check that numerical features are all in one chunk
                    assert sorted(contained_features) == sorted(numerical_features_list)

                    # check that ordering is exactly same as in channel spec - required for performance
                    assert contained_features == numerical_features_list

                    # check numerical dtype
                    for feature in contained_features:
                        assert np.dtype(self.feature_spec[feature][DTYPE_SELECTOR]) == np.float16

                elif first_feature == label_feature_name:
                    # check that label feature is in a separate file
                    assert len(contained_features) == 1

                    # check label dtype
                    assert np.dtype(self.feature_spec[first_feature][DTYPE_SELECTOR]) == np.bool

                else:
                    assert False, "Feature of unknown type"

            # check that all features appeared in mapping
            assert sorted(mapping_features) == sorted(feature_spec_features)


def get_embedding_sizes(fspec: FeatureSpec, max_table_size: Optional[int]) -> List[int]:
    if max_table_size is not None:
        return [min(s, max_table_size) for s in fspec.get_categorical_sizes()]
    else:
        return fspec.get_categorical_sizes()


def distribute_to_buckets(sizes: Sequence[int], buckets_num: int):
    def sum_sizes(indices):
        return sum(sizes[i] for i in indices)

    def argsort(sequence, reverse: bool = False):
        idx_pairs = [(x, i) for i, x in enumerate(sequence)]
        sorted_pairs = sorted(idx_pairs, key=lambda pair: pair[0], reverse=reverse)
        return [i for _, i in sorted_pairs]

    max_bucket_size = math.ceil(len(sizes) / buckets_num)
    idx_sorted = deque(argsort(sizes, reverse=True))
    buckets = [[] for _ in range(buckets_num)]
    final_buckets = []

    while idx_sorted:
        bucket = buckets[0]
        bucket.append(idx_sorted.popleft())

        if len(bucket) == max_bucket_size:
            final_buckets.append(buckets.pop(0))

        buckets.sort(key=sum_sizes)

    final_buckets += buckets

    return final_buckets


def get_device_mapping(embedding_sizes: Sequence[int], num_gpus: int = 1):
    """Get device mappings for hybrid parallelism

    Bottom MLP running on device 0. Embeddings will be distributed across among all the devices.

    Optimal solution for partitioning set of N embedding tables into K devices to minimize maximal subset sum
    is an NP-hard problem. Additionally, embedding tables distribution should be nearly uniform due to the performance
    constraints. Therefore, suboptimal greedy approach with max bucket size is used.

    Args:
        embedding_sizes (Sequence[int]): embedding tables sizes
        num_gpus (int): Default 8.

    Returns:
        device_mapping (dict):
    """
    gpu_buckets = distribute_to_buckets(embedding_sizes, num_gpus)

    vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]
    vectors_per_gpu[0] += 1  # count bottom mlp

    return {
        "bottom_mlp": 0,
        "embedding": gpu_buckets,
        "vectors_per_gpu": vectors_per_gpu,
    }
