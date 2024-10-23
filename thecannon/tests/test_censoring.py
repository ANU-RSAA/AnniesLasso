"""
Unit tests for `thecannon.censoring`.
"""

import pytest
import numpy as np

from ..censoring import Censors, create_mask

# Censors tests


@pytest.mark.parametrize(
    "label_names",
    [
        ("a", "b", "c", 2, 3, 4),
    ],
)
@pytest.mark.parametrize("num_pixels", [2, 4, 6, 8, 10, 100, 1000, 10000])
def test_censors_init(label_names, num_pixels):
    dummy_items = {l: np.ones(num_pixels, dtype=bool) for l in label_names}

    # This creation line implicitly tests Censors.__setitem__
    cn = Censors(label_names, num_pixels, items=dummy_items)

    # Check that everything has been set correctly
    assert cn._label_names == label_names
    assert cn._num_pixels == num_pixels
    assert cn.keys() == dummy_items.keys()
    for k in cn.keys():
        assert np.all(cn[k] == dummy_items[k])

    # Check the __getstate__ return
    gs = cn.__getstate__()
    assert gs["label_names"] == label_names
    assert gs["num_pixels"] == num_pixels
    assert gs["items"].keys() == dummy_items.keys()
    for k in gs["items"].keys():
        assert np.all(gs["items"][k] == dummy_items[k])

    # Check the property returns
    assert cn.label_names == label_names
    assert cn.num_pixels == num_pixels


class TestCensorsBadSetitem:

    cn = Censors(["x", "y", "z"], 100)

    @pytest.mark.parametrize("bad_label", [1, 2, 3, "a", "b", "c"])
    def test_censors_setitem_bad_label(self, bad_label):
        with pytest.raises(ValueError):
            self.cn[bad_label] = None

    @pytest.mark.parametrize(
        "bad_mask", [np.ones(n, dtype=bool) for n in [1, 10, 1000, 10000]]
    )
    def test_censors_setitem_bad_mask(self, bad_mask):
        with pytest.raises(ValueError):
            self.cn["x"] = bad_mask


@pytest.mark.parametrize(
    "a",
    [
        [],
        [{"args_key_1": np.ones(100, dtype=bool)}],
        [
            {
                "args_key_1": np.ones(100, dtype=bool),
                "args_key_2": np.zeros(100, dtype=bool),
            }
        ],
    ],
)
@pytest.mark.parametrize(
    "k",
    [
        {},
        {"kwargs_key_1": np.ones(100, dtype=bool)},
        {
            "kwargs_key_1": np.ones(100, dtype=bool),
            "kwargs_key_2": np.zeros(100, dtype=bool),
        },
    ],
)
def test_censors_update(a, k):
    label_list = (
        [_ for _ in a[0].keys()] + [_ for _ in k.keys()]
        if len(a) > 0
        else [_ for _ in k.keys()]
    )
    cn = Censors(label_list, 100)  # Set labels as per coming inputs, num_pixels = 100
    # import pdb; pdb.set_trace()

    cn.update(*a, **k)
    for key in label_list:
        if "kwargs" in key:
            assert np.all(cn[key] == k[key]), f"Failed to update key {key}"
        else:
            assert np.all(cn[key] == a[0][key]), f"Failed to update key {key}"


@pytest.mark.parametrize("a", [[{}, {}]])
def test_censors_update_multiargs(a):
    with pytest.raises(TypeError):
        cn = Censors(["a", "b"], 10)
        cn.update(*a)


@pytest.mark.parametrize("labels", [("a", "b", "c")])
@pytest.mark.parametrize(
    "default",
    [
        "a",
        "b",
        "c",
    ],
)
def test_censors_setdefault(labels, default):
    cn = Censors(labels, 10)
    x = cn.setdefault(default, value=np.ones(10))
    assert np.all(x == np.ones(10)), "Failed to set default correctly"


@pytest.mark.parametrize(
    "dispersion,censored_regions, expected_mask",
    [
        (np.ones(4), [], np.zeros(4, dtype=bool)),
        (
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            [(0.0, 0.15), (0.35, 0.45)],
            np.asarray([True, False, False, True, False]),
        ),
        (np.asarray([1, 2, 3]), [(0, 5)], np.ones(3, dtype=bool)),
        (np.asarray([0.2, 0.4, 0.6]), [0.3, 0.5], np.asarray([False, True, False])),  # Flat censored_region
    ],
)
def test_create_mask(dispersion, censored_regions, expected_mask):
    m = create_mask(dispersion, censored_regions)
    assert np.all(m == expected_mask), "Failed to create expected mask"


@pytest.mark.parametrize("dispersion,censored_regions,expected_error", [
    (np.ones(3), [0.1, 0.2, 0.3], ValueError),  # Too many values in flat censored_regions
    (np.ones(3), ["1", 2], ValueError),  # Bad value in flat censored_regions, first value
    (np.ones(3), [1, "2"], ValueError),  # Bad value in flat censored_regions, second value
    (np.ones(3), [(1, 2), ("2", 3)], ValueError), # Bad value in list of tuples (posn 1)
    (np.ones(3), [("1", 2), (2, 3)], ValueError), # Bad value in list of tuples (posn 1)
    (np.ones(3), [(1, 2), (2, "3")], ValueError), # Bad value in list of tuples (posn 2)
    (np.ones(3), [(1, "2"), (2, 3)], ValueError), # Bad value in list of tuples (posn 2)
])
def test_create_mask_bad(dispersion, censored_regions, expected_error):
    with pytest.raises(expected_error):
        _ = create_mask(dispersion, censored_regions)
