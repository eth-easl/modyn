import logging
import os
import sys
from pathlib import Path

import numpy as np
from modyn.utils import flatten, get_partition_for_worker

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

    def __init__(
        self,
        trigger_sample_directory: str,
    ):
        self.trigger_sample_directory = trigger_sample_directory
        if not Path(self.trigger_sample_directory).exists():
            Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created the trigger sample directory {self.trigger_sample_directory}.")
        if sys.maxsize < 2**63 - 1:
            raise RuntimeError("Modyn Selector Implementation requires a 64-bit system.")

    def get_trigger_samples(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        retrieval_worker_id: int = -1,
        total_retrieval_workers: int = -1,
        num_samples_trigger_partition: int = -1,
    ) -> list[tuple[int, float]]:
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
        :param num_samples_trigger_partition: the total number of samples per trigger and partition
        :return: the trigger samples
        """
        if not Path(self.trigger_sample_directory).exists():
            raise FileNotFoundError(f"The trigger sample directory {self.trigger_sample_directory} does not exist.")
        assert (retrieval_worker_id >= 0 and total_retrieval_workers >= 0) or (
            retrieval_worker_id < 0 and total_retrieval_workers < 2
        ), "Either both or none of the retrieval worker id must be negative and \
            the total retrieval workers must be smaller than 2."
        if retrieval_worker_id < 0 and total_retrieval_workers < 2:
            return self._get_all_samples(pipeline_id, trigger_id, partition_id)
        assert num_samples_trigger_partition > 0, "The number of samples per trigger must be positive."
        return self._get_worker_samples(
            pipeline_id,
            trigger_id,
            partition_id,
            retrieval_worker_id,
            total_retrieval_workers,
            num_samples_trigger_partition,
        )

    def _get_worker_samples(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        retrieval_worker_id: int,
        total_retrieval_workers: int,
        num_samples_trigger_partition: int,
    ) -> list[tuple[int, float]]:
        """
        Return the trigger samples for the given pipeline id, trigger id and partition id that are assigned to the
        retrieval worker.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param retrieval_worker_id: the id of the retrieval worker
        :param total_retrieval_workers: the total number of retrieval workers
        :param num_samples_trigger_partition: the total number of samples per trigger and partition
        :return: the trigger samples
        """
        start_index, worker_subset_size = get_partition_for_worker(
            retrieval_worker_id, total_retrieval_workers, num_samples_trigger_partition
        )

        current_index = 0

        triple_list: list[tuple[Path, int, int]] = []
        for file in sorted(os.listdir(self.trigger_sample_directory)):
            if file.startswith(f"{pipeline_id}_{trigger_id}_{partition_id}_"):
                file_path = Path(self.trigger_sample_directory) / file
                if current_index >= start_index + worker_subset_size:
                    #  We have already retrieved all the samples for the worker
                    break
                num_samples_in_file = self._get_num_samples_in_file(file_path)
                if current_index + num_samples_in_file <= start_index:
                    # The samples in the file are before the samples for the worker
                    current_index += num_samples_in_file
                    continue
                if current_index + num_samples_in_file < start_index + worker_subset_size:
                    # The head of samples for the worker are in the file, either partially from
                    # start_index - current_index to the end of the file if start_index > current_index
                    # or completely from 0 to the end of the file.
                    # Because the end index is exclusive, we compare < instead of <= otherwise we would retrieve
                    # one more sample than we should
                    triple_list.append(
                        (
                            file_path,
                            start_index - current_index if start_index - current_index >= 0 else 0,
                            num_samples_in_file,
                        )
                    )
                    current_index += num_samples_in_file
                    continue
                # We are at the tail of the file and the samples for the worker are in the file, either from
                #  the beginning if start_index - current_index < 0 or from start_index - current_index if the
                #  tail is in the same file as the head
                triple_list.append(
                    (
                        file_path,
                        start_index - current_index if start_index - current_index >= 0 else 0,
                        start_index + worker_subset_size - current_index,
                    )
                )
                break

        # We need to flatten the list of lists of np arrays and then reshape it to get the list of tuples
        return [
            (int(key), float(weight))  # type: ignore
            for (key, weight) in map(
                tuple,  # type: ignore
                np.array(
                    flatten(
                        [
                            self._parse_file_subset(file_path, start_index, end_index)
                            for file_path, start_index, end_index in triple_list
                        ]
                    )
                ),
            )
        ]

    def _get_all_samples(self, pipeline_id: int, trigger_id: int, partition_id: int) -> list[tuple[int, float]]:
        """
        Return all the samples for the given pipeline id, trigger id and partition id.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :return: the trigger samples
        """

        return [
            (int(key), float(weight))  # type: ignore
            for (key, weight) in map(
                tuple,  # type: ignore
                np.array(
                    flatten(
                        [
                            self._parse_file(Path(self.trigger_sample_directory) / file)
                            for file in sorted(os.listdir(self.trigger_sample_directory))
                            if file.startswith(f"{pipeline_id}_{trigger_id}_{partition_id}_")
                        ]
                    )
                ),
            )
        ]

    def save_trigger_sample(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        trigger_samples: np.ndarray,
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
        if trigger_samples.dtype != np.dtype("i8,f8"):
            raise ValueError(f"Unexpected dtype: {trigger_samples.dtype}\nExpected: {np.dtype('i8,f8')}")

        Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)

        samples_file = Path(self.trigger_sample_directory) / f"{pipeline_id}_{trigger_id}_{partition_id}_{insertion_id}"

        assert not Path(samples_file).exists(), (
            f"Trigger samples file {samples_file} already exists. " f"Please delete it if you want to overwrite it."
        )

        self._write_file(samples_file, trigger_samples)

    def _write_file(self, file_path: Path, trigger_samples: np.ndarray) -> None:
        """Write the trigger samples to the given file.

        Args:
            file_path (str): File path to write to.
            trigger_samples (list[tuple[int, float]]): List of trigger samples.
        """
        np.save(file_path, trigger_samples, allow_pickle=False, fix_imports=False)

    def _parse_file(self, file_path: Path) -> np.ndarray:
        """Parse the given file and return the samples.

        Args:
            file_path (str): File path to parse.

        Returns:
            list[tuple[int, float]]: List of trigger samples.
        """
        return np.load(file_path, allow_pickle=False, fix_imports=False)

    def _parse_file_subset(self, file_path: Path, start_index: int, end_index: int) -> np.memmap:
        """Parse the given file and return the samples. Only return samples between start_index
           inclusive and end_index exclusive.

        Args:
            file_path (str): File path to parse.
            end_index (int): The index of the last sample to return.

        Returns:
            list[tuple[int, float]]: List of trigger samples.
        """
        return np.load(file_path, allow_pickle=False, fix_imports=False, mmap_mode="r").take(
            range(start_index, end_index), axis=0
        )

    def _get_num_samples_in_file(self, file_path: Path) -> int:
        """Get the number of samples in the given file.

        Args:
            file_path (str): File path to parse.
        """
        return np.load(file_path, allow_pickle=False, fix_imports=False, mmap_mode="r").shape[0]
