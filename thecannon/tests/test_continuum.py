#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test continuum module.
"""

import pytest
import numpy as np

from thecannon import continuum


@pytest.mark.parametrize("L", ["a", [1, 3], np.asarray([1, 3]), 0, 0.0, 1e-16, -1e-16])
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


@pytest.mark.parametrize(
    "dispersion",
    [
        np.ones(3),
        np.ones((3, 2)),
        np.ones((2, 3, 4)),
        np.ones((5, 4, 3, 2)),
    ],
)
@pytest.mark.parametrize("L", [-1.0, 1.0])
@pytest.mark.parametrize("order", [2, 4, 6, 10])
def test__continuum_design_matrix_return(dispersion, L, order):
    matrix = continuum._continuum_design_matrix(dispersion, L, order)
    assert matrix.shape == (
        2 * order + 1,
        dispersion.size,
    ), "_continuum_design_matrix returned wrong shape"
