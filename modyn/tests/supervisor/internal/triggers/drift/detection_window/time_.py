from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    TimeWindowingStrategy,
)
from modyn.supervisor.internal.triggers.drift.detection_window.time_ import (
    TimeDetectionWindows,
)


def test_time_detection_window_manager_no_overlap() -> None:
    config = TimeWindowingStrategy(limit_cur="50s", limit_ref="100s", allow_overlap=False)
    assert config.limit_cur == "50s"
    assert config.limit_ref == "100s"
    assert not config.allow_overlap

    windows = TimeDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # partial fill current_
    windows.inform_data([(1, 15), (2, 30)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(1, 15), (2, 30)]

    # current_ overflow -> fill current_reservoir_
    windows.inform_data([(3, 45), (4, 60)])
    assert len(windows.current) == 4
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(1, 15), (2, 30), (3, 45), (4, 60)]

    # overflow current_ and current_reservoir_
    windows.inform_data([(5, 75), (6, 90), (7, 120)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 3
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(5, 75), (6, 90), (7, 120)]
    assert list(windows.current_reservoir) == [(2, 30), (3, 45), (4, 60)]

    # trigger: reset current_ and move data to reference_
    for _ in range(0, 3):
        # idempotency test
        windows.inform_trigger()
        assert len(windows.current) == 0
        assert len(windows.current_reservoir) == 0
        assert len(windows.reference) == 6
        assert len(windows.exclusive_current) == 0
        assert list(windows.reference) == [
            (2, 30),
            (3, 45),
            (4, 60),
            (5, 75),
            (6, 90),
            (7, 120),
        ]

    windows.inform_data([(8, 135)])
    assert len(windows.current) == 1
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 6
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(8, 135)]
    assert list(windows.reference) == [
        (2, 30),
        (3, 45),
        (4, 60),
        (5, 75),
        (6, 90),
        (7, 120),
    ]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 6
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [
        (3, 45),
        (4, 60),
        (5, 75),
        (6, 90),
        (7, 120),
        (8, 135),
    ]


def test_time_detection_window_manager_no_overlap_ref_smaller_cur() -> None:
    config = TimeWindowingStrategy(limit_cur="100s", limit_ref="50s", allow_overlap=False)
    assert config.limit_cur == "100s"
    assert config.limit_ref == "50s"
    assert not config.allow_overlap

    windows = TimeDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # fill current_
    windows.inform_data([(1, 15), (2, 30)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(1, 15), (2, 30)]

    # current_ overflow
    windows.inform_data([(3, 45), (4, 60), (5, 75), (6, 90), (7, 150)])
    assert len(windows.current) == 4
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(4, 60), (5, 75), (6, 90), (7, 150)]

    # trigger: reset current_ and move data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 1
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [(7, 150)]

    windows.inform_data([(8, 190), (9, 210)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 1
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(8, 190), (9, 210)]
    assert list(windows.reference) == [(7, 150)]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 2
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [(8, 190), (9, 210)]


def test_time_detection_window_manager_with_overlap() -> None:
    config = TimeWindowingStrategy(limit_cur="50s", limit_ref="100s", allow_overlap=True)
    assert config.limit_cur == "50s"
    assert config.limit_ref == "100s"
    assert config.allow_overlap

    windows = TimeDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # partial fill current_
    windows.inform_data([(1, 15), (2, 30)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(1, 15), (2, 30)]
    assert list(windows.exclusive_current) == [(1, 15), (2, 30)]

    # current_ overflow -> fill current_reservoir_
    windows.inform_data([(3, 45), (4, 60)])
    assert len(windows.current) == 4
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 4
    assert list(windows.current) == [(1, 15), (2, 30), (3, 45), (4, 60)]
    assert list(windows.exclusive_current) == [(1, 15), (2, 30), (3, 45), (4, 60)]

    # overflow current_ and current_reservoir_
    windows.inform_data([(5, 75), (6, 90), (7, 120)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 6
    assert list(windows.current) == [(5, 75), (6, 90), (7, 120)]
    assert list(windows.exclusive_current) == [
        (2, 30),
        (3, 45),
        (4, 60),
        (5, 75),
        (6, 90),
        (7, 120),
    ]

    # trigger: reset current_ and move data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 6
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(5, 75), (6, 90), (7, 120)]
    assert list(windows.reference) == [
        (2, 30),
        (3, 45),
        (4, 60),
        (5, 75),
        (6, 90),
        (7, 120),
    ]

    windows.inform_data([(8, 135)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 6
    assert len(windows.exclusive_current) == 1
    assert list(windows.current) == [(6, 90), (7, 120), (8, 135)]
    assert list(windows.exclusive_current) == [(8, 135)]
    assert list(windows.reference) == [
        (2, 30),
        (3, 45),
        (4, 60),
        (5, 75),
        (6, 90),
        (7, 120),
    ]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 6
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(6, 90), (7, 120), (8, 135)]
    assert list(windows.reference) == [
        (3, 45),
        (4, 60),
        (5, 75),
        (6, 90),
        (7, 120),
        (8, 135),
    ]


def test_time_detection_window_manager_with_overlap_ref_smaller_cur() -> None:
    config = TimeWindowingStrategy(limit_cur="100s", limit_ref="50s", allow_overlap=True)
    assert config.limit_cur == "100s"
    assert config.limit_ref == "50s"
    assert config.allow_overlap

    windows = TimeDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # fill current_
    windows.inform_data([(1, 15), (2, 30)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(1, 15), (2, 30)]
    assert list(windows.exclusive_current) == [(1, 15), (2, 30)]

    # current_ overflow
    windows.inform_data([(3, 45), (4, 60), (5, 75), (6, 90), (7, 150)])
    assert len(windows.current) == 4
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 1
    assert list(windows.current) == [(4, 60), (5, 75), (6, 90), (7, 150)]
    assert list(windows.exclusive_current) == [(7, 150)]

    # trigger: reset current_ and move data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 4
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 1
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(4, 60), (5, 75), (6, 90), (7, 150)]
    assert list(windows.reference) == [(7, 150)]

    windows.inform_data([(8, 190), (9, 210)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 1
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(7, 150), (8, 190), (9, 210)]
    assert list(windows.reference) == [(7, 150)]
    assert list(windows.exclusive_current) == [(8, 190), (9, 210)]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 2
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(7, 150), (8, 190), (9, 210)]
    assert list(windows.reference) == [(8, 190), (9, 210)]
