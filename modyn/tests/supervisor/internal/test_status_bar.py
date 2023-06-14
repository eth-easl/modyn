import enlighten
from modyn.supervisor.internal.supervisor_counter import CurrentEvent, SupervisorCounter


def test_counter_training_one_epoch():
    mgr = enlighten.get_manager()
    pbar = SupervisorCounter(mgr, 12, 50, 1)

    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.IDLE

    # batch of 25 elements
    pbar.progress_counter(25, 0, True)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 25
    assert pbar.sample_pbar.count == 25
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.TRAINING

    # batch of 25 elements
    pbar.progress_counter(50, 0, True)
    assert pbar.current_epoch == 1
    assert pbar.last_samples_training == 50
    assert pbar.current_event == CurrentEvent.IDLE
    assert pbar.sample_pbar is None  # COunter is closed
    assert pbar.last_samples_downsampling == 0


def test_counter_training_several_epochs():
    mgr = enlighten.get_manager()
    pbar = SupervisorCounter(mgr, 12, 50, 1)

    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.IDLE

    # batch of 25 elements
    pbar.progress_counter(25, 0, True)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 25
    assert pbar.sample_pbar.count == 25
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.TRAINING

    # batch of 25 elements
    pbar.progress_counter(50, 0, True)
    assert pbar.current_epoch == 1
    assert pbar.last_samples_training == 50
    assert pbar.current_event == CurrentEvent.IDLE
    assert pbar.sample_pbar is None  # COunter is closed
    assert pbar.last_samples_downsampling == 0

    # batch of 25 elements
    pbar.progress_counter(75, 0, True)
    assert pbar.current_epoch == 1
    assert pbar.last_samples_training == 75
    assert pbar.sample_pbar.count == 25
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.TRAINING

    # batch of 75 elements
    pbar.progress_counter(150, 0, True)
    assert pbar.current_epoch == 3
    assert pbar.last_samples_training == 150
    assert pbar.sample_pbar is None
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.IDLE

    # batch of 149 elements
    pbar.progress_counter(299, 0, True)
    assert pbar.current_epoch == 5
    assert pbar.last_samples_training == 299
    assert pbar.sample_pbar.count == 49
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.TRAINING

    # batch of 2 elements
    pbar.progress_counter(301, 0, True)
    assert pbar.current_epoch == 6
    assert pbar.last_samples_training == 301
    assert pbar.sample_pbar.count == 1
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.TRAINING


def test_counter_downsampling():
    mgr = enlighten.get_manager()
    pbar = SupervisorCounter(mgr, 12, 50, 1)

    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.IDLE

    pbar.progress_counter(0, 10, False)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.sample_pbar.count == 10
    assert pbar.last_samples_downsampling == 10
    assert pbar.current_event == CurrentEvent.DOWNSAMPLING

    pbar.progress_counter(0, 50, False)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.sample_pbar is None
    assert pbar.last_samples_downsampling == 50
    assert pbar.current_event == CurrentEvent.IDLE


def test_counter_alternating():
    mgr = enlighten.get_manager()
    pbar = SupervisorCounter(mgr, 12, 50, 1)

    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.last_samples_downsampling == 0
    assert pbar.current_event == CurrentEvent.IDLE

    # downsampling
    pbar.progress_counter(0, 10, False)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.sample_pbar.count == 10
    assert pbar.last_samples_downsampling == 10
    assert pbar.current_event == CurrentEvent.DOWNSAMPLING

    pbar.progress_counter(0, 50, False)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 0
    assert pbar.sample_pbar is None
    assert pbar.last_samples_downsampling == 50
    assert pbar.current_event == CurrentEvent.IDLE

    # training
    pbar.progress_counter(25, 0, True)
    assert pbar.current_epoch == 0
    assert pbar.last_samples_training == 25
    assert pbar.sample_pbar.count == 25
    assert pbar.last_samples_downsampling == 50
    assert pbar.current_event == CurrentEvent.TRAINING

    pbar.progress_counter(50, 0, True)
    assert pbar.current_epoch == 1
    assert pbar.last_samples_training == 50
    assert pbar.sample_pbar is None
    assert pbar.current_event == CurrentEvent.IDLE

    # new epoch
    pbar.progress_counter(75, 0, True)
    assert pbar.current_epoch == 1
    assert pbar.last_samples_training == 75
    assert pbar.sample_pbar.count == 25
    assert pbar.current_event == CurrentEvent.TRAINING

    pbar.progress_counter(100, 0, True)
    assert pbar.current_epoch == 2
    assert pbar.last_samples_training == 100
    assert pbar.sample_pbar is None
    assert pbar.current_event == CurrentEvent.IDLE

    # downsampling again

    pbar.progress_counter(0, 10, False)
    assert pbar.current_epoch == 2
    assert pbar.sample_pbar.count == 10
    assert pbar.last_samples_downsampling == 10
    assert pbar.current_event == CurrentEvent.DOWNSAMPLING

    pbar.progress_counter(0, 50, False)
    assert pbar.current_epoch == 2
    assert pbar.sample_pbar is None
    assert pbar.last_samples_downsampling == 50
    assert pbar.current_event == CurrentEvent.IDLE
