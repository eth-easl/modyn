import math
from typing import Any, Optional, Union

import apricot
import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)
from scipy.sparse import csr_matrix


class RemoteCRAIGDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    WIP
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        per_sample_loss: Any,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector)

        self.per_sample_loss_fct = per_sample_loss

        if "selection_type" not in params_from_selector:
            self.selection_type = "Supervised"
        else:
            self.selection_type = params_from_selector["selection_type"]
            assert self.selection_type in ["Supervised", "PerBatch"]

        if "optimizer" not in params_from_selector:
            self.optimizer = "lazy"
        else:
            self.optimizer = params_from_selector["optimizer"]
            # these are from Apricot, idk how much control we want to give here
            assert self.optimizer in [
                "random",
                "modular",
                "naive",
                "lazy",
                "approximate-lazy",
                "two-stage",
                "stochastic",
                "sample",
                "greedi",
                "bidirectional",
            ]

        self.g_is: list[torch.Tensor] = []
        self.dist_mat_size = 0

        self.number_of_samples_seen = 0

        # Alert: perBatch requires to store the indices of every batch
        self.batch_wise_indices: list[list] = []
        self.indices: list[int] = []

        self._samples_available = False

    def setup_sample_then_batch(self) -> None:
        assert self.sample_before_batch

    def accumulate_sample_then_batch(self, model_output: torch.Tensor, target: torch.Tensor, sample_ids: list) -> None:
        assert self.sample_before_batch
        loss = self.per_sample_loss_fct(model_output, target).sum()
        l0_grads = torch.autograd.grad(loss, model_output, retain_graph=False)[0]

        if self.selection_type == "PerBatch":
            self.batch_wise_indices.append(sample_ids)
            self.g_is.append(l0_grads.mean(dim=0).view(1, -1))
            self.dist_mat_size += 1
        else:
            self.indices += sample_ids
            self.g_is.append(l0_grads)
            self.dist_mat_size += model_output.size()[0]

        self.number_of_samples_seen += model_output.size()[0]

    def end_sample_then_batch(self) -> None:
        assert self.sample_before_batch
        self._samples_available = True

    def samples_available(self) -> bool:
        assert self.sample_before_batch
        return self._samples_available

    def get_samples(self) -> np.ndarray:
        assert self.sample_before_batch
        assert self._samples_available
        self._samples_available = False
        selected_ids, selected_weights = self.finalize_selection()
        samples_list = np.empty(len(selected_ids), dtype=np.dtype("i8,f8"))
        for i, _ in enumerate(selected_ids):
            samples_list[i] = (
                (selected_ids[i], selected_weights[i])
                if self.selection_type != "Supervised"
                else (self.indices[selected_ids[i]], selected_weights[i])
            )

        del self.indices
        del self.batch_wise_indices
        del self.g_is

        return samples_list

    def finalize_selection(self) -> tuple[list, torch.Tensor]:
        dist_mat = torch.zeros([self.dist_mat_size, self.dist_mat_size], dtype=torch.float32)

        if self.selection_type == "PerBatch":
            self.g_is = torch.cat(self.g_is, dim=0)
            dist_mat = self.distance(self.g_is, self.g_is).cpu()
        else:
            # Original code. Adapted to support multiple dataloaders.
            # first_i = True
            # for i, g_i in enumerate(self.g_is):
            #    if first_i:
            #        size_b = g_i.size(0)
            #        first_i = False
            #    for j, g_j in enumerate(self.g_is):
            #        distance = self.distance(g_i, g_j).cpu()
            #        dist_mat[
            #            i * size_b : i * size_b + g_i.size(0), j * size_b : j * size_b + g_j.size(0)
            #        ] = distance

            current_i = 0
            for g_i in self.g_is:
                current_j = 0
                for g_j in self.g_is:
                    end_i = current_i + g_i.shape[0]
                    end_j = current_j + g_j.shape[0]
                    dist_mat[current_i:end_i, current_j:end_j] = self.distance(g_i, g_j).cpu()
                    current_j = end_j
                current_i = end_i

        const = torch.max(dist_mat).item()
        dist_mat = (const - dist_mat).numpy()

        budget = (
            int(self.number_of_samples_seen * self.downsampled_batch_ratio / 100)
            if self.sample_before_batch
            else self.downsampled_batch_size
        )

        if self.selection_type == "Supervised":
            total_greedy_list, gammas = self._finalize_supervised(budget, dist_mat)
        else:
            total_greedy_list, gammas = self._finalize_per_batch(budget, dist_mat)

        total_greedy_list = [int(x) for x in total_greedy_list]
        return total_greedy_list, torch.Tensor(gammas)

    def _finalize_supervised(self, budget: int, dist_mat: torch.Tensor) -> tuple[list, list]:
        idxs = torch.arange(0, self.number_of_samples_seen).long()
        row = idxs.repeat_interleave(self.number_of_samples_seen)
        col = idxs.repeat(self.number_of_samples_seen)
        data = dist_mat.flatten()
        sparse_simmat = csr_matrix(
            (data, (row.numpy(), col.numpy())), shape=(self.number_of_samples_seen, self.number_of_samples_seen)
        )
        dist_mat = sparse_simmat
        facility_location = apricot.functions.facilityLocation.FacilityLocationSelection(
            random_state=0, metric="precomputed", n_samples=budget, optimizer=self.optimizer
        )
        sim_sub = facility_location.fit_transform(sparse_simmat)
        total_greedy_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
        gammas = self.compute_gamma(total_greedy_list, dist_mat)

        return total_greedy_list, gammas

    def _finalize_per_batch(self, budget: int, dist_mat: torch.Tensor) -> tuple[list, list]:
        facility_location = apricot.functions.facilityLocation.FacilityLocationSelection(
            random_state=0,
            metric="precomputed",
            n_samples=math.ceil(budget / self.batch_size),
            optimizer=self.optimizer,
            verbose=False,
        )
        sim_sub = facility_location.fit_transform(dist_mat)
        temp_list = list(np.array(np.argmax(sim_sub, axis=1)).reshape(-1))
        gammas_temp = self.compute_gamma(temp_list, dist_mat)

        total_greedy_list = []
        gammas = []
        for index, element in enumerate(temp_list):
            tmp = self.batch_wise_indices[element]
            total_greedy_list.extend(tmp)
            gammas.extend([gammas_temp[index]] * len(tmp))

        return total_greedy_list, gammas

    def distance(self, first: torch.Tensor, second: torch.Tensor, exp: Optional[int] = 2) -> torch.Tensor:
        first_size_0 = first.size(0)
        second_size_0 = second.size(0)
        first_size_1 = first.size(1)
        first = first.unsqueeze(1).expand(first_size_0, second_size_0, first_size_1)
        second = second.unsqueeze(0).expand(first_size_0, second_size_0, first_size_1)
        dist = torch.pow(first - second, exp).sum(2)
        return dist

    def compute_gamma(self, idxs: list, dist_mat: Union[torch.Tensor, csr_matrix]) -> list:
        assert self.selection_type in ["PerBatch", "Supervised"]

        gamma = [0 for _ in range(len(idxs))]
        best = dist_mat[idxs]  # .to(self.device)
        rep = np.argmax(best, axis=0)

        if self.selection_type == "PerBatch":
            for i in rep:
                gamma[i] += 1
        elif self.selection_type == "Supervised":
            for i in range(rep.shape[1]):
                gamma[rep[0, i]] += 1
        return gamma

    def get_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def batch_then_sample(
        self,
        forward_output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.sample_before_batch
        assert self.selection_type == "Supervised"

        loss = self.per_sample_loss_fct(forward_output, target).sum()
        l0_grads = torch.autograd.grad(loss, forward_output, retain_graph=False)[0]

        self.g_is = [l0_grads]
        self.dist_mat_size = forward_output.size()[0]

        self.number_of_samples_seen = forward_output.size()[0]

        selected_ids, selected_weights = self.finalize_selection()

        del self.g_is
        del self.dist_mat_size

        return torch.Tensor(selected_ids).int(), selected_weights
