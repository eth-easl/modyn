import pathlib
import logging
import torch
import math
from modyn.models import DLRM
from modyn.storage.internal.filesystem_wrapper.local_filesystem_wrapper import LocalFilesystemWrapper
from modyn.storage.internal.file_wrapper.binary_file_wrapper import BinaryFileWrapper
from tqdm import tqdm
import io
import numpy as np
from torch.utils.data import DataLoader


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

def instantiate_model(model_path):
    with open(model_path, "rb") as file:
        checkpoint = torch.load(io.BytesIO(file.read()))
        assert "model" in checkpoint

    dlrm = DLRM({"device": "cuda:0",
                 "embedding_dim": 128,
                "interaction_op": "cuda_dot",
                "hash_indices": False,
                "bottom_mlp_sizes": [512, 256, 128],
                "top_mlp_sizes": [1024, 1024, 512, 256, 1],
                "embedding_type": "joint_fused",
                "use_cpp_mlp": True,
                "fp16": False,
                "bottom_features_ordered": False
                })

    dlrm.load_state_dict(checkpoint["model"])
    return dlrm


class EvaluationDataset(torch.Dataset):
    def __init__(self, evaluation_data):
        self.fs_wrapper = LocalFilesystemWrapper(evaluation_data)
        files = self.fs_wrapper.list()

        self.file_wrappers = [BinaryFileWrapper(file, {"byteorder": "little", "recordsize": 160, "labelsize": 4, "file_extension": ".bin"}, self.fs_wrapper) for file in files]
        logger.info(f"Initialized {len(self.file_wrappers)} file wrappers for evaluation")

    def __len__(self):
        return sum([file_wrapper.get_number_of_samples() for file_wrapper in self.file_wrappers])

    def bytes_parser_function(x: bytes) -> dict:
        num_features = x[:52]
        cat_features = x[52:]
        num_features_array = np.frombuffer(num_features, dtype=np.float32)
        cat_features_array = np.frombuffer(cat_features, dtype=np.int32)
        
        return {
            "numerical_input": torch.asarray(num_features_array, copy=True, dtype=torch.float32),
            "categorical_input": torch.asarray(cat_features_array, copy=True, dtype=torch.int32)
        }

    def __getitem__(self, idx):
        # Inefficiently obtain correct file of sample index, we should really build an evaluation component...
        current_fw_idx = 0
        curr_sample_idx = 0

        while curr_sample_idx + (self.file_wrappers[current_fw_idx].get_number_of_samples() - 1) < idx: # if idx does not fall into range of curr_fw
            curr_sample_idx += self.file_wrappers[current_fw_idx].get_number_of_samples()
            current_fw_idx += 1

            if current_fw_idx >= len(self.file_wrappers):
                raise ValueError(f"Invalid idx: {idx}")

        file_wrapper = self.file_wrappers[current_fw_idx]
        idx_into_filewrapper = idx - curr_sample_idx

        sample_bytes = file_wrapper.get_sample(idx_into_filewrapper)
        sample_label = file_wrapper.get_label(idx_into_filewrapper)

        return self.bytes_parser_function(sample_bytes), sample_label


def evaluate_model(model_path: pathlib.Path, evaluation_data: pathlib.Path) -> dict:
    model = instantiate_model(model_path)
    model.to("cuda:0")
    model.eval()

    dataset = EvaluationDataset(evaluation_data)
    dataloader = DataLoader(dataset, batch_size=15360, num_workers=8)

    logger.info(f"Running evaluation on {len(dataset)} samples.")

    with torch.no_grad():
        y_true = []
        y_score = []

        for batch in tqdm(dataloader):
            data = batch[0].to("cuda:0")
            label = batch[1]
            output = model(data)
            y_true.append(label.cpu())
            y_score.append(output.cpu())

    y_true = torch.cat(y_true)
    y_score = torch.sigmoid(torch.cat(y_score)).float()
    auc = roc_auc_score(y_true, y_score)
    accuracy = acc_score(y_true, y_score)

    return { "acc": accuracy, "auc": auc }