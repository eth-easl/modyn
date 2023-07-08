import enlighten
from modyn.supervisor.internal.utils.training_status_tracker import CurrentEvent, TrainingStatusTracker


def test_counter_training_one_epoch():
    mgr = enlighten.get_manager()
    status_tracker = TrainingStatusTracker(mgr, 12, 50, 100)

    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.IDLE

    # batch of 25 elements
    status_tracker.progress_counter(25, 0, True)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 25
    assert status_tracker.sample_pbar.count == 25
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.TRAINING

    # batch of 25 elements
    status_tracker.progress_counter(50, 0, True)
    assert status_tracker.current_epoch == 1
    assert status_tracker.last_samples_training == 50
    assert status_tracker.current_event == CurrentEvent.IDLE
    assert status_tracker.sample_pbar is None  # COunter is closed
    assert status_tracker.last_samples_downsampling == 0


def test_counter_training_several_epochs():
    mgr = enlighten.get_manager()
    status_tracker = TrainingStatusTracker(mgr, 12, 50, 100)

    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.IDLE

    # batch of 25 elements
    status_tracker.progress_counter(25, 0, True)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 25
    assert status_tracker.sample_pbar.count == 25
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.TRAINING

    # batch of 25 elements
    status_tracker.progress_counter(50, 0, True)
    assert status_tracker.current_epoch == 1
    assert status_tracker.last_samples_training == 50
    assert status_tracker.current_event == CurrentEvent.IDLE
    assert status_tracker.sample_pbar is None  # COunter is closed
    assert status_tracker.last_samples_downsampling == 0

    # batch of 25 elements
    status_tracker.progress_counter(75, 0, True)
    assert status_tracker.current_epoch == 1
    assert status_tracker.last_samples_training == 75
    assert status_tracker.sample_pbar.count == 25
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.TRAINING

    # batch of 75 elements
    status_tracker.progress_counter(150, 0, True)
    assert status_tracker.current_epoch == 3
    assert status_tracker.last_samples_training == 150
    assert status_tracker.sample_pbar is None
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.IDLE

    # batch of 149 elements
    status_tracker.progress_counter(299, 0, True)
    assert status_tracker.current_epoch == 5
    assert status_tracker.last_samples_training == 299
    assert status_tracker.sample_pbar.count == 49
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.TRAINING

    # batch of 2 elements
    status_tracker.progress_counter(301, 0, True)
    assert status_tracker.current_epoch == 6
    assert status_tracker.last_samples_training == 301
    assert status_tracker.sample_pbar.count == 1
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.TRAINING


def test_counter_downsampling():
    mgr = enlighten.get_manager()
    status_tracker = TrainingStatusTracker(mgr, 12, 50, 100)

    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.IDLE

    status_tracker.progress_counter(0, 10, False)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.sample_pbar.count == 10
    assert status_tracker.last_samples_downsampling == 10
    assert status_tracker.current_event == CurrentEvent.DOWNSAMPLING

    status_tracker.progress_counter(0, 50, False)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.sample_pbar is None
    assert status_tracker.last_samples_downsampling == 50
    assert status_tracker.current_event == CurrentEvent.IDLE


def test_counter_alternating():
    mgr = enlighten.get_manager()
    status_tracker = TrainingStatusTracker(mgr, 12, 50, 100)

    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.last_samples_downsampling == 0
    assert status_tracker.current_event == CurrentEvent.IDLE

    # downsampling
    status_tracker.progress_counter(0, 10, False)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.sample_pbar.count == 10
    assert status_tracker.last_samples_downsampling == 10
    assert status_tracker.current_event == CurrentEvent.DOWNSAMPLING

    status_tracker.progress_counter(0, 50, False)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 0
    assert status_tracker.sample_pbar is None
    assert status_tracker.last_samples_downsampling == 50
    assert status_tracker.current_event == CurrentEvent.IDLE

    # training
    status_tracker.progress_counter(25, 0, True)
    assert status_tracker.current_epoch == 0
    assert status_tracker.last_samples_training == 25
    assert status_tracker.sample_pbar.count == 25
    assert status_tracker.last_samples_downsampling == 50
    assert status_tracker.current_event == CurrentEvent.TRAINING

    status_tracker.progress_counter(50, 0, True)
    assert status_tracker.current_epoch == 1
    assert status_tracker.last_samples_training == 50
    assert status_tracker.sample_pbar is None
    assert status_tracker.current_event == CurrentEvent.IDLE

    # new epoch
    status_tracker.progress_counter(75, 0, True)
    assert status_tracker.current_epoch == 1
    assert status_tracker.last_samples_training == 75
    assert status_tracker.sample_pbar.count == 25
    assert status_tracker.current_event == CurrentEvent.TRAINING

    status_tracker.progress_counter(100, 0, True)
    assert status_tracker.current_epoch == 2
    assert status_tracker.last_samples_training == 100
    assert status_tracker.sample_pbar is None
    assert status_tracker.current_event == CurrentEvent.IDLE

    # downsampling again

    status_tracker.progress_counter(0, 10, False)
    assert status_tracker.current_epoch == 2
    assert status_tracker.sample_pbar.count == 10
    assert status_tracker.last_samples_downsampling == 10
    assert status_tracker.current_event == CurrentEvent.DOWNSAMPLING

    status_tracker.progress_counter(0, 50, False)
    assert status_tracker.current_epoch == 2
    assert status_tracker.sample_pbar is None
    assert status_tracker.last_samples_downsampling == 50
    assert status_tracker.current_event == CurrentEvent.IDLE
