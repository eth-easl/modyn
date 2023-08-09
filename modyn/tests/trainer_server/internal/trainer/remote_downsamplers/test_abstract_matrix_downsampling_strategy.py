# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
)


def get_sampler_config():
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False, "args": {}}
    return 0, 0, 0, params_from_selector, per_sample_loss_fct


@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_init():
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config())

    assert amds.requires_coreset_methods_support
    assert not amds.matrix_elements
    assert amds.matrix_content is None


@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_collect_embeddings():
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config())

    amds.matrix_content = MatrixContent.EMBEDDINGS

    assert amds.requires_coreset_methods_support
    assert not amds.matrix_elements  # thank you pylint! amds.matrix_elements == []

    first_embedding = torch.randn((4, 5))
    second_embedding = torch.randn((3, 5))
    amds.inform_samples([1, 2, 3, 4], None, None, first_embedding)
    amds.inform_samples([21, 31, 41], None, None, second_embedding)

    assert np.concatenate(amds.matrix_elements).shape == (7, 5)
    assert all(torch.equal(el1, el2) for el1, el2 in zip(amds.matrix_elements, [first_embedding, second_embedding]))
    assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]

    third_embedding = torch.randn((23, 5))
    amds.inform_samples(list(range(1000, 1023)), None, None, third_embedding)

    assert np.concatenate(amds.matrix_elements).shape == (30, 5)
    assert all(
        torch.equal(el1, el2)
        for el1, el2 in zip(amds.matrix_elements, [first_embedding, second_embedding, third_embedding])
    )
    assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41] + list(range(1000, 1023))


@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_collect_gradients():
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config())
    amds.matrix_content = MatrixContent.GRADIENTS

    first_output = torch.randn((4, 2))
    first_output.requires_grad = True
    first_target = torch.tensor([1, 1, 1, 0])
    first_embedding = torch.randn((4, 5))
    amds.inform_samples([1, 2, 3, 4], first_output, first_target, first_embedding)

    second_output = torch.randn((3, 2))
    second_output.requires_grad = True
    second_target = torch.tensor([0, 1, 0])
    second_embedding = torch.randn((3, 5))
    amds.inform_samples([21, 31, 41], second_output, second_target, second_embedding)

    assert len(amds.matrix_elements) == 2

    # expected shape = (a,b)
    # a = 7 (4 samples in the first batch and 3 samples in the second batch)
    # b = 5 * 2 + 2 where 5 is the input dimension of the last layer and 2 is the output one
    assert np.concatenate(amds.matrix_elements).shape == (7, 12)

    assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]
