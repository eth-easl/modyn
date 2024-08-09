from unittest.mock import MagicMock, patch

import enlighten

from modynclient.client.internal.utils import EvaluationStatusTracker


def test_evaluation_status_tracker_init():
    evaluation_tracker = EvaluationStatusTracker(dataset_id="EVAL", dataset_size=1000)

    assert evaluation_tracker.dataset_id == "EVAL"
    assert evaluation_tracker.dataset_size == 1000
    assert evaluation_tracker.counter is None
    assert evaluation_tracker.last_samples_seen == 0


def test_create_counter():
    mgr = enlighten.get_manager()

    evaluation_tracker = EvaluationStatusTracker(dataset_id="EVAL", dataset_size=1000)
    evaluation_tracker.create_counter(mgr, 10, 1)

    assert evaluation_tracker.counter is not None
    assert evaluation_tracker.counter.total == 1000
    assert evaluation_tracker.counter.count == 0
    assert evaluation_tracker.counter.desc == "[Training 10] Evaluation 1 on dataset EVAL"


@patch.object(EvaluationStatusTracker, "end_counter")
def test_progress_counter(test_end_counter: MagicMock):
    evaluation_tracker = EvaluationStatusTracker(dataset_id="EVAL", dataset_size=1000)
    mgr = enlighten.get_manager()
    evaluation_tracker.create_counter(mgr, 10, 1)

    evaluation_tracker.progress_counter(500)
    assert evaluation_tracker.counter.total == 1000
    assert evaluation_tracker.counter.count == 500
    test_end_counter.assert_not_called()

    evaluation_tracker.progress_counter(1000)

    assert evaluation_tracker.last_samples_seen == 1000
    test_end_counter.assert_called_once_with(False)


def test_end_counter_valid():
    evaluation_tracker = EvaluationStatusTracker(dataset_id="EVAL", dataset_size=1000)
    mgr = enlighten.get_manager()
    evaluation_tracker.create_counter(mgr, 10, 1)

    evaluation_tracker.end_counter(False)

    assert evaluation_tracker.counter.count == 1000


def test_end_counter_invalid():
    evaluation_tracker = EvaluationStatusTracker(dataset_id="EVAL", dataset_size=1000)
    mgr = enlighten.get_manager()
    evaluation_tracker.create_counter(mgr, 10, 1)

    evaluation_tracker.end_counter(True)

    assert evaluation_tracker.counter.count == 0
