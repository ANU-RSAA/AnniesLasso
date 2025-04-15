"""
Unit tests for `thecannon.censoring`.
"""

import pytest
import numpy as np
from copy import copy, deepcopy

from ..censoring import Censors, create_mask  # , design_matrix_mask
from ..vectorizer.base import BaseVectorizer

# Censors tests


@pytest.mark.parametrize(
    "label_names",
    [
        ("a", "b", "c", 2, 3, 4),
    ],
)
@pytest.mark.parametrize("num_pixels", [2, 4, 6, 8, 10, 100, 1000, 10000])
class TestCensorInitAndEq:
    def test_censors_init_and_eq(self, label_names, num_pixels):
        dummy_items = {l: np.ones(num_pixels, dtype=bool) for l in label_names}

        # This creation line implicitly tests Censors.__setitem__
        cn = Censors(label_names, num_pixels, items=dummy_items)

        # Check that everything has been set correctly
        assert cn._label_names == list(label_names)
        assert cn._num_pixels == num_pixels
        assert cn.keys() == dummy_items.keys()
        for k in cn.keys():
            assert np.all(cn[k] == dummy_items[k])

        # Check the __getstate__ return
        gs = cn.__getstate__()
        assert gs["label_names"] == list(label_names)
        assert gs["num_pixels"] == num_pixels
        assert gs["items"].keys() == dummy_items.keys()
        for k in gs["items"].keys():
            assert np.all(gs["items"][k] == dummy_items[k])

        # Check the property returns
        assert cn.label_names == list(label_names)
        assert cn.num_pixels == num_pixels

        # Check that an identically-created Censor is considered equal
        cn2 = Censors(label_names=label_names, num_pixels=num_pixels, items=dummy_items)
        assert cn == cn2, "__eq__ not returning True as expected"

        # Check copies and deepcopies match
        cn3 = copy(cn)
        assert cn == cn3 and cn3 == cn, "__eq__ not saying copy are equal"

        cn4 = deepcopy(cn)
        assert cn == cn4 and cn4 == cn, "__eq__ not saying deepcopy are equal"

    def test_censors_neq_label_names(self, label_names, num_pixels):
        cn1 = Censors(label_names, num_pixels, items={l: np.ones(num_pixels, dtype=bool) for l in label_names})

        # Rearrange label_names
        cn2 = Censors(label_names[-1:] + label_names[1:], num_pixels, items={l: np.ones(num_pixels, dtype=bool) for l in label_names[-1:] + label_names[1:]})
        assert cn1 != cn2, "__eq__ did not pick up on rearranged label_names list"

        # Drop a label name
        cn3 = Censors(label_names[1:], num_pixels, items={l: np.ones(num_pixels, dtype=bool) for l in label_names[1:]})
        assert cn1 != cn2, "__eq__ did not pick up on different length label_names list"

    def test_censors_neq_num_pixels(self, label_names, num_pixels):
        cn1 = Censors(label_names, num_pixels, items={l: np.ones(num_pixels, dtype=bool) for l in label_names})
        cn2 = Censors(label_names, num_pixels + 1, items={l: np.ones(num_pixels+1, dtype=bool) for l in label_names})

        assert cn1 != cn2, "__eq__ did not pick up on different num_pixels"

    def test_censors_neq_items(self, label_names, num_pixels):
        cn1 = Censors(label_names, num_pixels, items={l: np.ones(num_pixels, dtype=bool) for l in label_names})
        cn2 = Censors(label_names, num_pixels, items={l: np.zeros(num_pixels, dtype=bool) for l in label_names})

        assert cn1 != cn2, "__eq__ did not pick up on different masks in Censor"


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
        (
            np.asarray([0.2, 0.4, 0.6]),
            [0.3, 0.5],
            np.asarray([False, True, False]),
        ),  # Flat censored_region
    ],
)
def test_create_mask(dispersion, censored_regions, expected_mask):
    m = create_mask(dispersion, censored_regions)
    assert np.all(m == expected_mask), "Failed to create expected mask"


@pytest.mark.parametrize(
    "dispersion,censored_regions,expected_error",
    [
        (
            np.ones(3),
            [0.1, 0.2, 0.3],
            ValueError,
        ),  # Too many values in flat censored_regions
        (
            np.ones(3),
            ["1", 2],
            ValueError,
        ),  # Bad value in flat censored_regions, first value
        (
            np.ones(3),
            [1, "2"],
            ValueError,
        ),  # Bad value in flat censored_regions, second value
        (
            np.ones(3),
            [(1, 2), ("2", 3)],
            ValueError,
        ),  # Bad value in list of tuples (posn 1)
        (
            np.ones(3),
            [("1", 2), (2, 3)],
            ValueError,
        ),  # Bad value in list of tuples (posn 1)
        (
            np.ones(3),
            [(1, 2), (2, "3")],
            ValueError,
        ),  # Bad value in list of tuples (posn 2)
        (
            np.ones(3),
            [(1, "2"), (2, 3)],
            ValueError,
        ),  # Bad value in list of tuples (posn 2)
    ],
)
def test_create_mask_bad(dispersion, censored_regions, expected_error):
    with pytest.raises(expected_error):
        _ = create_mask(dispersion, censored_regions)


# @pytest.mark.parametrize("censor", ["a", 1, 1.6, BaseVectorizer(["a"], [[("a", 1)]])])
# @pytest.mark.parametrize("vectorizer", [BaseVectorizer(["a"], [[("a", 1)]])])
# def test_design_matrix_mask_bad_censor(censor, vectorizer):
#     with pytest.raises(TypeError):
#         _ = design_matrix_mask(censor, vectorizer)


# @pytest.mark.parametrize("vectorizer", ["a", 1, 1.6, Censors(["a"], 10)])
# @pytest.mark.parametrize("censor", [Censors(["a"], 10)])
# def test_design_matrix_mask_bad_vectorizer(censor, vectorizer):
#     with pytest.raises(TypeError):
#         _ = design_matrix_mask(censor, vectorizer)
