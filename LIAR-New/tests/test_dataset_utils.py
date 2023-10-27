import pytest

from dataset_merge_utils import build_index


@pytest.fixture()
def example_data_entries():
    return [
        {"example_index": 1, "example_value": "1"},
        {"example_index": 0, "example_value": "0"},
    ]


def test_build_index(example_data_entries):
    """
    Verify that items in the index are placed correctly.

    Ordering is not important for this use case.
    """
    index = build_index(example_data_entries, "example_index")

    assert index == {
        1: {"example_index": 1, "example_value": "1"},
        0: {"example_index": 0, "example_value": "0"},
    }
