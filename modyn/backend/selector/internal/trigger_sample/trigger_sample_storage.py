"""
A trigger sample is a tuple of (sample_id, sample_weight) that is used to select a sample for a trigger.

The sample_id is the id of the sample in the database. The sample_weight is the weight of the sample in the trigger.

This class is used to store and retrieve trigger samples from the local file system. The trigger samples are stored
in the directory specified in the modyn config file. The file name is the concatenation of the pipeline id, the
trigger id and the partition id. The file contains one line per sample. Each line contains the sample id and the
sample weight separated by a comma.
"""
import logging
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def get_trigger_samples(
    pipeline_id: int,
    trigger_id: int,
    partition_id: int,
    trigger_sample_directory: str,
    retrieval_worker_id: int = -1,
    total_retrieval_workers: int = -1,
    num_samples_trigger: int = -1,
) -> List[Tuple[int, float]]:
    """
    Return the trigger samples for the given pipeline id, trigger id and partition id.

    If the retrieval worker id and the total retrieval workers are negative, then we are not using the parallel
    retrieval of samples. In this case, we just return all the samples.

    If the retrieval worker id and the total retrieval workers are positive, then we are using the parallel
    retrieval of samples. In this case, we return the samples that are assigned to the retrieval worker.

    :param pipeline_id: the id of the pipeline
    :param trigger_id: the id of the trigger
    :param partition_id: the id of the partition
    :param trigger_sample_directory: the directory where the trigger samples are stored
    :param retrieval_worker_id: the id of the retrieval worker
    :param total_retrieval_workers: the total number of retrieval workers
    :param total_samples: the total number of samples
    :return: the trigger samples
    """
    Path(trigger_sample_directory).mkdir(parents=True, exist_ok=True)

    if retrieval_worker_id < 0 and total_retrieval_workers < 0:
        #  If the retrieval worker id and the total retrieval workers are negative, then we are not using
        #  the parallel retrieval of samples. In this case, we just return all the samples.
        samples = []
        for file in os.listdir(trigger_sample_directory):
            if file.startswith(f"{pipeline_id}_{trigger_id}_{partition_id}") and file.endswith(".txt"):
                with open(os.path.join(trigger_sample_directory, file), "r", encoding="utf-8") as file:
                    samples.extend(tuple(map(float, line.split(":"))) for line in file.readlines())
        return samples

    start_index, worker_subset_size = get_training_set_partition(
        retrieval_worker_id, total_retrieval_workers, num_samples_trigger
    )

    samples = []
    current_index = 0
    for file in os.listdir(trigger_sample_directory):
        if file.startswith(f"{pipeline_id}_{trigger_id}_{partition_id}") and file.endswith(".txt"):
            with open(os.path.join(trigger_sample_directory, file), "r", encoding="utf-8") as file:
                for line in file.readlines():
                    if start_index <= current_index < start_index + worker_subset_size:
                        samples.append(tuple(map(float, line.split(":"))))
                    current_index += 1
                    if current_index >= start_index + worker_subset_size:
                        break

    return samples


def get_training_set_partition(worker_id: int, total_workers: int, number_training_samples: int) -> Tuple[int, int]:
    """
    Return the required subset of training samples for the particular worker id
    The subset is calculated by taking an offset from the start based on the given worker id.

    If there is excess data (say there are 14 data points and 5 workers), there are at most
    num_workers extra samples. As such, we make each worker take on one extra, and the final
    worker takes on (probably less) the rest of the data. So we would have the first 4 take
    3 each and the last one takes 2.

    Returns:
        start_index: The index of the first sample to be used by the worker
        worker_subset_size: The number of samples to be used by the worker
    """
    if worker_id < 0 or worker_id >= total_workers:
        raise ValueError(f"Asked for worker id {worker_id}, but only have {total_workers} workers!")

    training_set_size = number_training_samples
    worker_subset_size = int(training_set_size / total_workers)

    if training_set_size % total_workers > 0:
        worker_subset_size += 1
        start_index = worker_id * worker_subset_size
        if worker_id == total_workers - 1:
            worker_subset_size = training_set_size - (worker_subset_size * (total_workers - 1))
    else:
        start_index = worker_id * worker_subset_size

    return start_index, worker_subset_size


def save_trigger_sample(
    pipeline_id: int,
    trigger_id: int,
    partition_id: int,
    trigger_samples: List[Tuple[int, float]],
    insertion_id: int,
    trigger_sample_directory: str,
) -> None:
    """
    Save the trigger samples for the given pipeline id, trigger id and partition id.

    :param pipeline_id: the id of the pipeline
    :param trigger_id: the id of the trigger
    :param partition_id: the id of the partition
    :param trigger_samples: the trigger samples
    :param insertion_id: the id of the insertion
    :param trigger_sample_directory: the directory where the trigger samples are stored
    """
    Path(trigger_sample_directory).mkdir(parents=True, exist_ok=True)

    samples_file = os.path.join(
        trigger_sample_directory, f"{pipeline_id}_{trigger_id}_{partition_id}_{insertion_id}.txt"
    )
    with open(samples_file, "w", encoding="utf-8") as file:
        file.write("\n".join(f"{x[0]}:{x[1]}" for x in trigger_samples))
