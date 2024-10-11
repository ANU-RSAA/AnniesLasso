"""
Unit tests for `thecannon.censoring`.
"""
import pytest
import numpy as np

from ..censoring import Censors

# Censors tests
   
@pytest.mark.parametrize("label_names", [("a", "b", "c", 2, 3, 4),])
@pytest.mark.parametrize("num_pixels", [2, 4, 6, 8, 10, 100, 1000, 10000])
def test_censors_init(label_names, num_pixels):
    dummy_items = {l: np.ones(num_pixels, dtype=bool) for l in label_names}
    c = Censors(label_names, num_pixels, items=dummy_items)
    assert c._label_names == label_names
    assert c._num_pixels == num_pixels
    assert c.keys() == dummy_items.keys()
    for k in c.keys():
        assert np.all(c[k] == dummy_items[k])
