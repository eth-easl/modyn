import time
from typing import Generator

from modyn.utils.timer import timed_generator


def test_timed_generator() -> None:
    def gen(r: int) -> Generator[int, None, None]:
        time.sleep(0.01)
        for i in range(r):
            time.sleep(0.05)
            yield i

    s0 = time.time()
    g = timed_generator(gen(3))
    s1 = time.time()
    assert s1 - s0 < 1e-3

    n = next(g)
    s2 = time.time()
    # should take ~60ms
    assert n[0] == 0 and abs((s2 - s1) * 1000 - 10 - 50) < 10 and abs(n[1] - 60) < 10

    n = next(g)
    s3 = time.time()
    # should take ~50ms
    assert n[0] == 1 and abs((s3 - s2) * 1000 - 50) < 10 and abs(n[1] - 50) < 10

    n = next(g)
    s4 = time.time()
    # should take ~50ms
    assert n[0] == 2 and abs((s4 - s3) * 1000 - 50) < 10 and abs(n[1] - 50) < 10
