import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.remote_loss_downsample import RemoteLossDownsampling


def test_sample_shape():
    model = torch.nn.Linear(10, 2)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    params_from_trainer = {"model": model, "per_sample_loss_fct": per_sample_loss_fct}
    sampler = RemoteLossDownsampling(params_from_selector, params_from_trainer)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    ids = list(range(8))

    sampled_data, weights, sampled_target, ids = sampler.sample(data, target, ids)

    assert sampled_data.shape[0] == downsampled_batch_size
    assert sampled_data.shape[1] == data.shape[1]
    assert weights.shape[0] == downsampled_batch_size
    assert sampled_target.shape[0] == downsampled_batch_size


def test_sample_weights():
    model = torch.nn.Linear(10, 2)
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    params_from_trainer = {"model": model, "per_sample_loss_fct": per_sample_loss_fct}
    sampler = RemoteLossDownsampling(params_from_selector, params_from_trainer)

    data = torch.randn(8, 10)
    target = torch.randint(2, size=(8,))
    ids = list(range(8))
    _, weights, _, ids = sampler.sample(data, target, ids)

    assert weights.sum() > 0
    assert set(ids) <= set(list(range(8)))


# Create a model that always predicts the same class
class AlwaysZeroModel(torch.nn.Module):
    def forward(self, data):
        return torch.zeros(data.shape[0])


def test_sample_loss_dependent_sampling():
    model = AlwaysZeroModel()
    downsampled_batch_size = 5
    per_sample_loss_fct = torch.nn.MSELoss(reduction="none")

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size}
    params_from_trainer = {"model": model, "per_sample_loss_fct": per_sample_loss_fct}
    sampler = RemoteLossDownsampling(params_from_selector, params_from_trainer)

    # Create a target with two classes, where half have a true label of 0 and half have a true label of 1
    target = torch.cat([torch.zeros(4), torch.ones(4)])

    # Create a data tensor with four points that have a loss of zero and four points that have a non-zero loss
    data = torch.cat([torch.randn(4, 10), torch.randn(4, 10)], dim=0)

    ids = list(range(8))
    _, _, sampled_target, _ = sampler.sample(data, target, ids)

    # Assert that no points with a loss of zero were selected
    assert (sampled_target == 0).sum() == 0
    assert (sampled_target > 0).sum() > 0
