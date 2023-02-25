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
from typing import Sequence


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
