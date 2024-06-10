import torch
from modyn.config import ModynConfig
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_uncertainty_downsampling_strategy import (
    RemoteUncertaintyDownsamplingStrategy,
)


def get_sampler_config(modyn_config: ModynConfig, balance=False):
    downsampling_ratio = 50
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {
        "downsampling_ratio": downsampling_ratio,
        "sample_then_batch": False,
        "args": {},
        "balance": balance,
        "score_metric": "LeastConfidence",
    }
    return 0, 0, 0, params_from_selector, modyn_config.model_dump(by_alias=True), per_sample_loss_fct, "cpu"


def test_init(dummy_system_config: ModynConfig):
    amds = RemoteUncertaintyDownsamplingStrategy(*get_sampler_config(dummy_system_config))

    assert not amds.requires_coreset_supporting_module
    assert not amds.scores
    assert not amds.index_sampleid_map
    assert not amds.requires_data_label_by_label


def test_collect_scores(dummy_system_config: ModynConfig):
    amds = RemoteUncertaintyDownsamplingStrategy(*get_sampler_config(dummy_system_config))
    with torch.inference_mode(mode=(not amds.requires_grad)):
        first_output = torch.randn((4, 5))
        second_output = torch.randn((3, 5))
        amds.inform_samples([1, 2, 3, 4], None, first_output, None, None)
        assert len(amds.scores) == 4
        amds.inform_samples([21, 31, 41], None, second_output, None, None)
        assert len(amds.scores) == 7

        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]

        third_output = torch.randn((23, 5))
        amds.inform_samples(list(range(1000, 1023)), None, third_output, None, None)

        assert len(amds.scores) == 30
        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41] + list(range(1000, 1023))


def test_collect_embedding_balance(dummy_system_config: ModynConfig):
    amds = RemoteUncertaintyDownsamplingStrategy(*get_sampler_config(dummy_system_config, True))
    with torch.inference_mode(mode=(not amds.requires_grad)):
        first_output = torch.randn((4, 5))
        second_output = torch.randn((3, 5))
        amds.inform_samples([1, 2, 3, 4], None, first_output, None, None)
        assert len(amds.scores) == 4
        amds.inform_samples([21, 31, 41], None, second_output, None, None)
        assert len(amds.scores) == 7

        assert amds.index_sampleid_map == [1, 2, 3, 4, 21, 31, 41]

        amds.inform_end_of_current_label()
        assert len(amds.already_selected_ids) == 3
        assert len(amds.already_selected_weights) == 3
        assert len(amds.scores) == 0
        assert len(amds.index_sampleid_map) == 0

        third_output = torch.randn((23, 5))
        amds.inform_samples(list(range(1000, 1023)), None, third_output, None, None)

        assert len(amds.scores) == 23
        assert amds.index_sampleid_map == list(range(1000, 1023))

        amds.inform_end_of_current_label()
        assert len(amds.already_selected_ids) == 14
        assert len(amds.already_selected_weights) == 14
        assert len(amds.scores) == 0
        assert len(amds.index_sampleid_map) == 0
