import io
import logging
import math
import pathlib
from typing import Optional

import numpy as np
import torch

# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
from modyn.models import DLRM
from modyn.storage.internal.file_wrapper.binary_file_wrapper import BinaryFileWrapper
from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

# Taken from https://github.com/NVIDIA/DeepLearningExamples/blob/678b470fd78e0fdb84b3173bc25164d766e7821f/PyTorch/Recommendation/DLRM/dlrm/scripts/utils.py#L289
def roc_auc_score(y_true, y_score):
    device = y_true.device
    y_true.squeeze_()
    y_score.squeeze_()
    if y_true.shape != y_score.shape:
        raise TypeError(f"Shape of y_true and y_score must match. Got {y_true.shape()} and {y_score.shape()}.")
    desc_score_indices = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = torch.nonzero(y_score[1:] - y_score[:-1], as_tuple=False).squeeze()
    threshold_idxs = torch.cat([distinct_value_indices, torch.tensor([y_true.numel() - 1], device=device)])
    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = torch.cat([torch.zeros(1, device=device), tps])
    fps = torch.cat([torch.zeros(1, device=device), fps])
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    area = torch.trapz(tpr, fpr).item()
    if math.isnan(area):
        return 0
    return area

def acc_score(y_true, y_score):
    y_pred = torch.ge(y_score, 0.5)
    return (torch.sum(y_pred == y_true) / y_pred.shape[0]).item()

def instantiate_model(model_path: Optional[pathlib.Path]):
    dlrm = DLRM(
            {
                "embedding_dim": 128,
                "interaction_op": "cuda_dot",
                "hash_indices": False,
                "bottom_mlp_sizes": [512, 256, 128],
                "top_mlp_sizes": [1024, 1024, 512, 256, 1],
                "embedding_type": "joint_fused",
                "num_numerical_features": 13,
                "use_cpp_mlp": True,
                "categorical_features_info": {
                    "cat_0": 7912889,
                    "cat_1": 33823,
                    "cat_2": 17139,
                    "cat_3": 7339,
                    "cat_4": 20046,
                    "cat_5": 4,
                    "cat_6": 7105,
                    "cat_7": 1382,
                    "cat_8": 63,
                    "cat_9": 5554114,
                    "cat_10": 582469,
                    "cat_11": 245828,
                    "cat_12": 11,
                    "cat_13": 2209,
                    "cat_14": 10667,
                    "cat_15": 104,
                    "cat_16": 4,
                    "cat_17": 968,
                    "cat_18": 15,
                    "cat_19": 8165896,
                    "cat_20": 2675940,
                    "cat_21": 7156453,
                    "cat_22": 302516,
                    "cat_23": 12022,
                    "cat_24": 97,
                    "cat_25": 35,
                }
            },
            device = "cuda:0", amp = False
    )

    logger.info("model instantiated")

    if model_path is not None:
        with open(model_path, "rb") as file:
            checkpoint = torch.load(io.BytesIO(file.read()))
            assert "model" in checkpoint

        logger.info("checkpoint loaded")

        dlrm.model.load_state_dict(checkpoint["model"])
        logger.info("model loaded")
    else:
        logger.warning("no checkpoint supplide")

    return dlrm


class EvaluationDataset(torch.utils.data.IterableDataset):
    def __init__(self, evaluation_data):
        logger.info(evaluation_data)

        self.fs_wrapper = LocalFilesystemWrapper(str(evaluation_data))
        files = self.fs_wrapper.list(str(evaluation_data))

        self.file_wrappers = [BinaryFileWrapper(str(evaluation_data / file), {"byteorder": "little", "record_size": 160, "label_size": 4, "file_extension": ".bin"}, self.fs_wrapper) for file in files]
        self.num_wrappers = len(self.file_wrappers)
        logger.info(f"Initialized {len(self.file_wrappers)} file wrappers for evaluation")

    def __len__(self):
        return sum([file_wrapper.get_number_of_samples() for file_wrapper in self.file_wrappers])

    def bytes_parser_function(self, x: bytes) -> dict:
        num_features = x[:52]
        cat_features = x[52:]
        num_features_array = np.frombuffer(num_features, dtype=np.float32)
        cat_features_array = np.frombuffer(cat_features, dtype=np.int32)

        return {
            "numerical_input": torch.asarray(num_features_array, copy=True, dtype=torch.float32),
            "categorical_input": torch.asarray(cat_features_array, copy=True, dtype=torch.long)
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = 0
            end_idx = self.num_wrappers
        else:
            num_per_worker = int(self.num_wrappers/worker_info.num_workers)
            worker_id = worker_info.id
            start_idx = worker_id*num_per_worker
            end_idx = (worker_id+1)*num_per_worker if worker_id < worker_info.num_workers-1 else self.num_wrappers

        logger.info(f"Worker: {worker_id}, start_idx: {start_idx}, end_idx: {end_idx}")
        for file_wrapper in self.file_wrappers[start_idx:end_idx]:
            num_samples = file_wrapper.get_number_of_samples()
            samples_bytes = file_wrapper.get_samples(0, num_samples)
            samples_labels = file_wrapper.get_all_labels()
            for sample, label in zip(samples_bytes, samples_labels):
                yield self.bytes_parser_function(sample), label


def evaluate_model(model_path: Optional[pathlib.Path], evaluation_data: pathlib.Path) -> dict:

    dlrm = instantiate_model(model_path)
    dlrm.model.eval()

    dataset = EvaluationDataset(evaluation_data)
    dataloader = DataLoader(dataset, batch_size=15360, num_workers=8)

    logger.info(f"Running evaluation on {len(dataset)} samples.")

    with torch.no_grad():
        y_true = []
        y_score = []

        for batch in tqdm(dataloader):
            data = {}
            for name, tensor in batch[0].items():
                data[name] = tensor.to("cuda:0")

            label = batch[1]
            output = dlrm.model(data)
            y_true.append(label.cpu())
            y_score.append(output.cpu())

    y_true = torch.cat(y_true)
    y_score = torch.sigmoid(torch.cat(y_score)).float()
    auc = roc_auc_score(y_true, y_score)
    accuracy = acc_score(y_true, y_score)

    return { "acc": accuracy, "auc": auc }