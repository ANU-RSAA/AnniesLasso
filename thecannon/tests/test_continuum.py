#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test continuum module.
"""

import pytest
import numpy as np

from thecannon import continuum


@pytest.mark.parametrize("L", ["a", [1, 3], np.asarray([1, 3])])
def test__continuum_design_matrix_bad_L(L):
    with pytest.raises(ValueError):
        continuum._continuum_design_matrix(np.asarray([]), L, 1)


@pytest.mark.parametrize(
    "dispersion",
    [
        [1, [2, 3]],  # Inhomogeneous shape
    ],
)
def test__continuum_design_matrix_bad_dispersion(dispersion):
    with pytest.raises(ValueError):
        continuum._continuum_design_matrix(dispersion, 1.0, 1)


@pytest.mark.parametrize("order", [-1, 0])
def test__continuum_design_matrix_bad_order(order):
    with pytest.raises(ValueError):
        continuum._continuum_design_matrix([], 1.0, order)
