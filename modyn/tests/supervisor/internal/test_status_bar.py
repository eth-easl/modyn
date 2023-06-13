import enlighten
from modyn.supervisor.internal.supervisor_counter import CurrentEvent, SupervisorCounter


def test_counter():
    mgr = enlighten.get_manager()
    pbar = SupervisorCounter(mgr, 12)

    pbar.start_training(samples_per_epoch=64)

    # first epoch, 15 samples
    pbar.progress_counter(15)
    assert pbar.current_epoch == 0
    assert pbar.last_samples == 15
    assert pbar.current_event == CurrentEvent.TRAINING

    # no new samples
    pbar.progress_counter(15)
    assert pbar.current_epoch == 0
    assert pbar.last_samples == 15
    assert pbar.current_event == CurrentEvent.TRAINING

    # 63/64 samples, still first epoch
    pbar.progress_counter(63)
    assert pbar.current_epoch == 0
    assert pbar.last_samples == 63
    assert pbar.current_event == CurrentEvent.TRAINING

    # finish the first epoch, start the second one
    pbar.progress_counter(64)
    assert pbar.current_epoch == 1
    assert pbar.last_samples == 64
    assert pbar.current_event == CurrentEvent.TRAINING

    # 3 new epochs in a row
    pbar.progress_counter(256)
    assert pbar.current_epoch == 4
    assert pbar.last_samples == 256
    assert pbar.current_event == CurrentEvent.TRAINING

    # 4 new epochs in a row
    pbar.progress_counter(511)
    assert pbar.current_epoch == 7
    assert pbar.last_samples == 511
    assert pbar.current_event == CurrentEvent.TRAINING
