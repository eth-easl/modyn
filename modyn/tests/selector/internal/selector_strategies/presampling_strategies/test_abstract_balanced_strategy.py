from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_balanced_strategy import (
    get_fair_share,
    get_fair_share_predicted_total,
)


def test_get_fair_share():
    # required less than available
    assert get_fair_share(1000, [10, 15, 30]) == 333
    assert get_fair_share_predicted_total(333, [10, 15, 30]) == 55

    # required more than base fs
    assert get_fair_share(100, [30, 50, 40, 28]) == 25
    assert get_fair_share_predicted_total(25, [30, 50, 40, 28]) == 100

    # one class below
    assert get_fair_share(100, [30, 50, 40, 21]) == 26
    assert get_fair_share_predicted_total(26, [30, 50, 40, 21]) == 99
    assert get_fair_share(100, [30, 50, 40, 20]) == 26
    assert get_fair_share_predicted_total(26, [30, 50, 40, 20]) == 98
    assert get_fair_share(100, [30, 50, 40, 19]) == 27
    assert get_fair_share_predicted_total(27, [30, 50, 40, 19]) == 100

    # two classes below
    assert get_fair_share(200, [30, 80, 90, 28]) == 71  # 30 + 71 + 71 + 28 = 200
    assert get_fair_share_predicted_total(71, [30, 80, 90, 28]) == 200
