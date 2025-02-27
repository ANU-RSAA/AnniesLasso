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
        [[1, ], ]  # Too many dimensions
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
        np.ones(30),
        np.ones(300),
        np.ones(int(1e4))
    ],
)
@pytest.mark.parametrize("L", [-1.0, 1.0, 1400])
@pytest.mark.parametrize("order", [2, 4, 6, 10])
def test__continuum_design_matrix_return(dispersion, L, order):
    matrix = continuum._continuum_design_matrix(dispersion, L, order)
    assert matrix.shape == (
        2 * order + 1,
        dispersion.size,
    ), "_continuum_design_matrix returned wrong shape"

@pytest.mark.parametrize("flux", [
    10,
    100,
    1000
])
@pytest.mark.parametrize("ivar", [
    10,
    100,
    1000
])
@pytest.mark.parametrize("dispersion", [
    10,
    100,
    1000
])
def test_sines_and_cosines_mismatched_inputs(flux, ivar, dispersion):
    if flux == ivar == dispersion:
        pytest.skip()
    