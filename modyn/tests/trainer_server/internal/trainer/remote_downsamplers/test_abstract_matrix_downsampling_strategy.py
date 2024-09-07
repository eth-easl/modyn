# pylint: disable=abstract-class-instantiated,unused-argument
from unittest.mock import patch

import numpy as np
import pytest
import torch

from modyn.config import ModynConfig
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
)


def get_sampler_config(
    dummy_system_config: ModynConfig, balance=False, matrix_content=MatrixContent.LAST_TWO_LAYERS_GRADIENTS
):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "args": {},
        "balance": balance,
        "ratio_max": 100,
    }
    return (
        0,
        0,
        0,
        params_from_selector,
        dummy_system_config.model_dump(by_alias=True),
        per_sample_loss_fct,
        "cpu",
        matrix_content,
    )


@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_init(dummy_system_config: ModynConfig):
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config(dummy_system_config))

    assert amds.requires_coreset_supporting_module
    assert not amds.matrix_elements
    assert amds.matrix_content == MatrixContent.LAST_TWO_LAYERS_GRADIENTS


@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_collect_embeddings(dummy_system_config: ModynConfig):
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    amds.matrix_content = MatrixContent.EMBEDDINGS
    with torch.inference_mode(mode=(not amds.requires_grad)):
        assert amds.requires_coreset_supporting_module
        assert not amds.matrix_elements  # thank you pylint! amds.matrix_elements == []

        first_embedding = torch.randn((4, 5))
        second_embedding = torch.randn((3, 5))
        amds.inform_samples([1, 2, 3, 4], None, torch.randn((4, 2)), None, first_embedding)
        amds.inform_samples([21, 31, 41], None, torch.randn((3, 2)), None, second_embedding)

        assert np.concatenate(amds.matrix_elements).shape == (7, 5)
        assert all(torch.equal(el1, el2) for el1, el2 in zip(amds.matrix_elements, [first_embedding, second_embedding]))
        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]

        third_embedding = torch.randn((23, 5))
        amds.inform_samples(list(range(1000, 1023)), None, torch.randn((23, 2)), None, third_embedding)

        assert np.concatenate(amds.matrix_elements).shape == (30, 5)
        assert all(
            torch.equal(el1, el2)
            for el1, el2 in zip(amds.matrix_elements, [first_embedding, second_embedding, third_embedding])
        )
        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41] + list(range(1000, 1023))


@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
@patch.object(
    AbstractMatrixDownsamplingStrategy, "_select_indexes_from_matrix", return_value=([0, 2], torch.Tensor([1.0, 3.0]))
)
def test_collect_embedding_balance(test_amds, dummy_system_config: ModynConfig):
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config(dummy_system_config, True))
    amds.matrix_content = MatrixContent.EMBEDDINGS
    with torch.inference_mode(mode=(not amds.requires_grad)):
        assert amds.requires_coreset_supporting_module
        assert amds.requires_data_label_by_label
        assert not amds.matrix_elements  # thank you pylint! amds.matrix_elements == []

        first_embedding = torch.randn((4, 5))
        second_embedding = torch.randn((3, 5))
        amds.inform_samples([1, 2, 3, 4], None, torch.randn((4, 2)), None, first_embedding)
        amds.inform_samples([21, 31, 41], None, torch.randn((3, 2)), None, second_embedding)

        assert np.concatenate(amds.matrix_elements).shape == (7, 5)
        assert all(torch.equal(el1, el2) for el1, el2 in zip(amds.matrix_elements, [first_embedding, second_embedding]))
        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]

        amds.inform_end_of_current_label()

        third_embedding = torch.randn((23, 5))
        assert len(amds.matrix_elements) == 0
        amds.inform_samples(list(range(1000, 1023)), None, torch.randn((23, 2)), None, third_embedding)

        assert np.concatenate(amds.matrix_elements).shape == (23, 5)
        assert all(torch.equal(el1, el2) for el1, el2 in zip(amds.matrix_elements, [third_embedding]))
        assert amds.index_sampleid_map == list(range(1000, 1023))
        assert amds.already_selected_samples == [1, 3]
        amds.inform_end_of_current_label()
        assert amds.already_selected_samples == [1, 3, 1000, 1002]


@pytest.mark.parametrize(
    "matrix_content", [MatrixContent.LAST_LAYER_GRADIENTS, MatrixContent.LAST_TWO_LAYERS_GRADIENTS]
)
@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_collect_gradients(matrix_content, dummy_system_config: ModynConfig):
    amds = AbstractMatrixDownsamplingStrategy(*get_sampler_config(dummy_system_config, matrix_content=matrix_content))
    with torch.inference_mode(mode=(not amds.requires_grad)):
        forward_input = torch.randn((4, 5))
        first_output = torch.randn((4, 2))
        first_output.requires_grad = True
        first_target = torch.tensor([1, 1, 1, 0])
        first_embedding = torch.randn((4, 5))
        amds.inform_samples([1, 2, 3, 4], forward_input, first_output, first_target, first_embedding)

        second_output = torch.randn((3, 2))
        second_output.requires_grad = True
        second_target = torch.tensor([0, 1, 0])
        second_embedding = torch.randn((3, 5))
        amds.inform_samples([21, 31, 41], forward_input, second_output, second_target, second_embedding)

        assert len(amds.matrix_elements) == 2

        # expected shape = (a, gradient_shape)
        # a = 7 (4 samples in the first batch and 3 samples in the second batch)
        if matrix_content == MatrixContent.LAST_LAYER_GRADIENTS:
            # shape same as the last dimension of output
            gradient_shape = 2
        else:
            # 5 is the input dimension of the last layer and 2 is the output one
            gradient_shape = 5 * 2 + 2
        assert np.concatenate(amds.matrix_elements).shape == (7, gradient_shape)

        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]


@pytest.mark.parametrize(
    "matrix_content", [MatrixContent.LAST_LAYER_GRADIENTS, MatrixContent.LAST_TWO_LAYERS_GRADIENTS]
)
@patch.multiple(AbstractMatrixDownsamplingStrategy, __abstractmethods__=set())
def test_collect_gradients_binary(matrix_content, dummy_system_config: ModynConfig):
    per_sample_loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
    sampler_config = list(get_sampler_config(dummy_system_config, matrix_content=matrix_content))
    sampler_config[5] = per_sample_loss_fct
    sampler_config = tuple(sampler_config)
    amds = AbstractMatrixDownsamplingStrategy(*sampler_config)
    with torch.inference_mode(mode=(not amds.requires_grad)):
        forward_input = torch.randn((4, 5))
        first_output = torch.randn((4,))
        first_output.requires_grad = True
        first_target = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
        first_embedding = torch.randn((4, 5))
        amds.inform_samples([1, 2, 3, 4], forward_input, first_output, first_target, first_embedding)

        second_output = torch.randn((3,))
        second_output.requires_grad = True
        second_target = torch.tensor([0, 1, 0], dtype=torch.float32)
        second_embedding = torch.randn((3, 5))
        amds.inform_samples([21, 31, 41], forward_input, second_output, second_target, second_embedding)

        assert len(amds.matrix_elements) == 2

        # expected shape = (a, gradient_shape)
        # a = 7 (4 samples in the first batch and 3 samples in the second batch)
        if matrix_content == MatrixContent.LAST_LAYER_GRADIENTS:
            # shape same as the last dimension of output
            gradient_shape = 1
        else:
            # 5 is the input dimension of the last layer and 1 is the output one
            gradient_shape = 5 * 1 + 1
        assert np.concatenate(amds.matrix_elements).shape == (7, gradient_shape)

        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]
