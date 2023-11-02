from typing import Any, Callable, Optional


def flatten(non_flat_list: list[list[Any]]) -> list[Any]:
    return [item for sublist in non_flat_list for item in sublist]


def get_partition_for_worker(worker_id: int, total_workers: int, total_num_elements: int) -> tuple[int, int]:
    """
    Returns the subset of data for a specific worker.
    This method splits the range of all elements evenly among all workers. If you e.g have 13 elements and want to split
    it among 5 workers, then workers [0, 1, 2] get 3 keys whereas workers [3, 4] get two keys.

    Args:
        worker_id: the id of the worker.
        total_workers: total amount of workers.
        total_num_elements: total amount of elements to split among the workers.

    Returns:
        tuple[int, int]: the start index (offset) and the total subset size.
    """
    if worker_id < 0 or worker_id >= total_workers:
        raise ValueError(f"Asked for worker id {worker_id}, but only have {total_workers} workers!")

    subset_size = int(total_num_elements / total_workers)
    worker_subset_size = subset_size

    threshold = total_num_elements % total_workers
    if threshold > 0:
        if worker_id < threshold:
            worker_subset_size += 1
            start_index = worker_id * (subset_size + 1)
        else:
            start_index = threshold * (subset_size + 1) + (worker_id - threshold) * subset_size
    else:
        start_index = worker_id * subset_size
    if start_index >= total_num_elements:
        start_index = 0

    return start_index, worker_subset_size
