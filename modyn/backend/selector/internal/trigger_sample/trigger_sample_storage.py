import logging
import os
from pathlib import Path
from typing import List, Tuple
from sys import getsizeof
import struct

logger = logging.getLogger(__name__)

class TriggerSampleStorage:
    """
    A trigger sample is a tuple of (sample_id, sample_weight) that is used to select a sample for a trigger.

    The sample_id is the id of the sample in the database. The sample_weight is the weight of the sample in the trigger.

    This class is used to store and retrieve trigger samples from the local file system. The trigger samples are stored
    in the directory specified in the modyn config file. The file name is the concatenation of the pipeline id, the
    trigger id and the partition id. The file contains one line per sample. Each line contains the sample id and the
    sample weight separated by a comma.
    """
    def __init__(self, trigger_sample_directory: str):
        self.trigger_sample_directory = trigger_sample_directory
        self.int_size = struct.calcsize("i")
        self.float_size = struct.calcsize("f")

    def get_trigger_samples(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
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
        :param retrieval_worker_id: the id of the retrieval worker
        :param total_retrieval_workers: the total number of retrieval workers
        :param total_samples: the total number of samples
        :return: the trigger samples
        """
        Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)

        if retrieval_worker_id < 0 and total_retrieval_workers < 0:
            return self._get_all_samples(pipeline_id, trigger_id, partition_id)
        else:
            return self._get_worker_samples(
                pipeline_id, trigger_id, partition_id, retrieval_worker_id, total_retrieval_workers, num_samples_trigger
            )

    def _get_worker_samples(self, pipeline_id: int, trigger_id: int, partition_id: int, retrieval_worker_id: int, total_retrieval_workers: int, num_samples_trigger: int) -> List[Tuple[int, float]]:
        """
        Return the trigger samples for the given pipeline id, trigger id and partition id that are assigned to the
        retrieval worker.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param retrieval_worker_id: the id of the retrieval worker
        :param total_retrieval_workers: the total number of retrieval workers
        :param total_samples: the total number of samples
        :return: the trigger samples
        """
        samples: List[Tuple[int, float]] = []

        start_index, worker_subset_size = self.get_training_set_partition(
            retrieval_worker_id, total_retrieval_workers, num_samples_trigger
        )

        current_index = 0
        for file in os.listdir(self.trigger_sample_directory):
            if file.startswith(f"{pipeline_id}_{trigger_id}_{partition_id}"):
                if current_index >= start_index + worker_subset_size:
                    break
                file_size = self._get_file_size(file)
                if current_index < start_index:
                    current_index += file_size
                    continue
                if current_index + file_size < start_index + worker_subset_size:
                    samples.extend(self._parse_file(file))
                    current_index += file_size
                    continue
                else:
                    samples.extend(self._parse_file_subset(file, start_index + worker_subset_size - current_index))
                    break
        return samples

    def _get_all_samples(self, pipeline_id: int, trigger_id: int, partition_id: int) -> List[Tuple[int, float]]:
        """
        Return all the samples for the given pipeline id, trigger id and partition id.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :return: the trigger samples
        """
        return [self._parse_file(file) for file in os.listdir(self.trigger_sample_directory) if file.startswith(f"{pipeline_id}{trigger_id}{partition_id}")]

    @staticmethod
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

        if start_index >= training_set_size:
            start_index = 0
            worker_subset_size = 0

        return start_index, worker_subset_size


    def save_trigger_sample(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        trigger_samples: List[Tuple[int, float]],
        insertion_id: int,
    ) -> None:
        """
        Save the trigger samples for the given pipeline id, trigger id and partition id.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param trigger_samples: the trigger samples
        :param insertion_id: the id of the insertion
        """
        Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)

        samples_file = os.path.join(
            self.trigger_sample_directory, f"{pipeline_id}_{trigger_id}_{partition_id}_{insertion_id}"
        )
        self._write_file(samples_file, trigger_samples)

    def _write_file(self, file_path: str, trigger_samples: List[Tuple[int, float]]) -> None:
        """Write the trigger samples to the given file.

        Args:
            file_path (str): File path to write to.
            trigger_samples (List[Tuple[int, float]]): List of trigger samples.
        """
        header = struct.pack("i", len(trigger_samples))

        with open(file_path, "wb") as file:
            file.write(header)
            [file.write(struct.pack("i", x[0]) + struct.pack("f", x[1])) for x in trigger_samples]

    def _parse_file(self, file_path: str) -> List[Tuple[int, float]]:
        """Parse the given file and return the samples.

        Args:
            file_path (str): File path to parse.

        Returns:
            List[Tuple[int, float]]: List of trigger samples.
        """
        with open(file_path, "rb") as file:
            header = struct.unpack("i", file.read(self.int_size))[0]
            samples = [(struct.unpack("i", file.read(self.int_size))[0], struct.unpack("f", file.read(self.float_size))[0]) for _ in range(header)]
            return samples
    
    def _parse_file_subset(self, file_path: str, end_index: int) -> List[Tuple[int, float]]:
        """Parse the given file and return the samples. We only return the first end_index samples.
        
        Args:
            file_path (str): File path to parse.
            end_index (int): The index of the last sample to return.

        Returns:
            List[Tuple[int, float]]: List of trigger samples.
        """
        with open(file_path, "rb") as file:
            file.read(self.int_size)
            samples = [(struct.unpack("i", file.read(self.int_size))[0], struct.unpack("f", file.read(self.float_size))[0]) for _ in range(end_index)]
            return samples[0:end_index]
    
    def _get_file_size(self, file_path: str) -> int:
        """Get the number of samples in the given file.
        
        Args:
            file_path (str): File path to parse.
        """
        with open(file_path, "rb") as file:
            header = struct.unpack("i", file.read(self.int_size))[0]
            return header
