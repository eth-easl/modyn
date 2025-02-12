from modyn.common.example_extension.example_extension import ExampleExtension


def test_init():
    ExampleExtension()


def test_sum_list():
    ext = ExampleExtension()
    empty_list = []
    assert ext.sum_list(empty_list) == 0

    one_list = [1]
    assert ext.sum_list(one_list) == 1

    another_list = list(range(10))
    assert ext.sum_list(another_list) == sum(another_list)

    another_list = list(range(100))
    assert ext.sum_list(another_list) == sum(another_list)
