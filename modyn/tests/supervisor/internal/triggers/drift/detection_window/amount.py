from modyn.config.schema.pipeline.trigger.drift.detection_window import AmountWindowingStrategy
from modyn.supervisor.internal.triggers.drift.detection_window.amount import AmountDetectionWindows


def test_amount_detection_window_manager_no_overlap() -> None:
    config = AmountWindowingStrategy(amount_cur=3, amount_ref=5, allow_overlap=False)
    assert config.amount_cur == 3
    assert config.amount_ref == 5
    assert not config.allow_overlap

    windows = AmountDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # partial fill current_
    windows.inform_data([(1, 100), (2, 200)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(1, 100), (2, 200)]

    # current_ overflow -> fill current_reservoir_
    windows.inform_data([(3, 300), (4, 400)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 1
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(2, 200), (3, 300), (4, 400)]
    assert list(windows.current_reservoir) == [(1, 100)]

    # overflow current_ and current_reservoir_
    windows.inform_data([(5, 500), (6, 600)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 2
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(4, 400), (5, 500), (6, 600)]
    assert list(windows.current_reservoir) == [(2, 200), (3, 300)]

    # trigger: reset current_ and move data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 5
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [
        (2, 200),
        (3, 300),
        (4, 400),
        (5, 500),
        (6, 600),
    ]

    windows.inform_data([(7, 700), (8, 800)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 5
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(7, 700), (8, 800)]
    assert list(windows.reference) == [
        (2, 200),
        (3, 300),
        (4, 400),
        (5, 500),
        (6, 600),
    ]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 5
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [
        (4, 400),
        (5, 500),
        (6, 600),
        (7, 700),
        (8, 800),
    ]


def test_amount_detection_window_manager_no_overlap_ref_smaller_cur() -> None:
    config = AmountWindowingStrategy(amount_cur=5, amount_ref=3, allow_overlap=False)
    assert config.amount_cur == 5
    assert config.amount_ref == 3
    assert not config.allow_overlap

    windows = AmountDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # partial fill current_
    windows.inform_data([(1, 100), (2, 200)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(1, 100), (2, 200)]

    # current_ overflow
    windows.inform_data([(3, 300), (4, 400), (5, 500), (6, 600)])
    assert len(windows.current) == 5
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(2, 200), (3, 300), (4, 400), (5, 500), (6, 600)]

    # trigger: reset current_ and move data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 3
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [(4, 400), (5, 500), (6, 600)]

    windows.inform_data([(7, 700), (8, 800)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 3
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(7, 700), (8, 800)]
    assert list(windows.reference) == [(4, 400), (5, 500), (6, 600)]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 3
    assert len(windows.exclusive_current) == 0
    assert list(windows.reference) == [(6, 600), (7, 700), (8, 800)]


def test_amount_detection_window_manager_with_overlap() -> None:
    config = AmountWindowingStrategy(amount_cur=3, amount_ref=5, allow_overlap=True)
    assert config.amount_cur == 3
    assert config.amount_ref == 5
    assert config.allow_overlap

    windows = AmountDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # partial fill current_
    windows.inform_data([(1, 100), (2, 200)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(1, 100), (2, 200)]
    assert list(windows.exclusive_current) == [(1, 100), (2, 200)]

    # current_ overflow
    windows.inform_data([(3, 300), (4, 400)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 4
    assert list(windows.current) == [(2, 200), (3, 300), (4, 400)]
    assert list(windows.exclusive_current) == [(1, 100), (2, 200), (3, 300), (4, 400)]

    # overflow current_ and exclusive_current
    windows.inform_data([(5, 500), (6, 600)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 5
    assert list(windows.current) == [(4, 400), (5, 500), (6, 600)]
    assert list(windows.exclusive_current) == [
        (2, 200),
        (3, 300),
        (4, 400),
        (5, 500),
        (6, 600),
    ]

    # trigger: DONT reset current_ but copy data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 5
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(4, 400), (5, 500), (6, 600)]
    assert list(windows.reference) == [
        (2, 200),
        (3, 300),
        (4, 400),
        (5, 500),
        (6, 600),
    ]

    windows.inform_data([(7, 700), (8, 800)])
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 5
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(6, 600), (7, 700), (8, 800)]
    assert list(windows.reference) == [
        (2, 200),
        (3, 300),
        (4, 400),
        (5, 500),
        (6, 600),
    ]
    assert list(windows.exclusive_current) == [(7, 700), (8, 800)]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 3
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 5
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(6, 600), (7, 700), (8, 800)]
    assert list(windows.reference) == [
        (4, 400),
        (5, 500),
        (6, 600),
        (7, 700),
        (8, 800),
    ]


def test_amount_detection_window_manager_with_overlap_ref_smaller_cur() -> None:
    config = AmountWindowingStrategy(amount_cur=5, amount_ref=3, allow_overlap=True)
    assert config.amount_cur == 5
    assert config.amount_ref == 3
    assert config.allow_overlap

    windows = AmountDetectionWindows(config)
    assert len(windows.current) == 0
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 0

    # partial fill current_
    windows.inform_data([(1, 100), (2, 200)])
    assert len(windows.current) == 2
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(1, 100), (2, 200)]
    assert list(windows.exclusive_current) == [(1, 100), (2, 200)]

    # current_ overflow
    windows.inform_data([(3, 300), (4, 400)])
    assert len(windows.current) == 4
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 3
    assert list(windows.current) == [(1, 100), (2, 200), (3, 300), (4, 400)]
    assert list(windows.exclusive_current) == [(2, 200), (3, 300), (4, 400)]

    # overflow current_ and exclusive_current
    windows.inform_data([(5, 500), (6, 600)])
    assert len(windows.current) == 5
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 0
    assert len(windows.exclusive_current) == 3
    assert list(windows.current) == [(2, 200), (3, 300), (4, 400), (5, 500), (6, 600)]
    assert list(windows.exclusive_current) == [(4, 400), (5, 500), (6, 600)]

    # trigger: DONT reset current_ but copy data to reference_
    windows.inform_trigger()
    assert len(windows.current) == 5
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 3
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(2, 200), (3, 300), (4, 400), (5, 500), (6, 600)]
    assert list(windows.reference) == [(4, 400), (5, 500), (6, 600)]

    windows.inform_data([(7, 700), (8, 800)])
    assert len(windows.current) == 5
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 3
    assert len(windows.exclusive_current) == 2
    assert list(windows.current) == [(4, 400), (5, 500), (6, 600), (7, 700), (8, 800)]
    assert list(windows.reference) == [(4, 400), (5, 500), (6, 600)]
    assert list(windows.exclusive_current) == [(7, 700), (8, 800)]

    # test ref overflow
    windows.inform_trigger()
    assert len(windows.current) == 5
    assert len(windows.current_reservoir) == 0
    assert len(windows.reference) == 3
    assert len(windows.exclusive_current) == 0
    assert list(windows.current) == [(4, 400), (5, 500), (6, 600), (7, 700), (8, 800)]
    assert list(windows.reference) == [(6, 600), (7, 700), (8, 800)]
