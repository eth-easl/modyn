import time
from collections.abc import Generator
from typing import TypeVar

X = TypeVar("X")


def timed_generator(generator: Generator[X, None, None]) -> Generator[tuple[X, float], None, None]:
    """Wrap a lazy generator and yields tuples of (item, elapsed_time).

    Args:
        generator: The lazy generator to wrap.

    Returns:
        A generator that yields (item, elapsed_time_millis) for each item in the original generator.
    """
    start_time = time.time()  # first evaluation starts when loop is entered9
    for item in generator:
        yield item, (time.time() - start_time) * 1000  # yield item and compute elapsed time
        start_time = time.time()  # next item requested: start timer again
