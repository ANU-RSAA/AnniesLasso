"""
Unit tests for `thecannon.fitting`.
"""

import pytest
import numpy as np
from thecannon import fitting


@pytest.mark.parametrize("argnone", ["theta", "design_matrix", "flux", "ivar"])
def test_chisq_input_none(argnone):
    dummy_array = np.ones((1,))
    args = [
        None if k == argnone else dummy_array
        for k in ["theta", "design_matrix", "flux", "ivar"]
    ]
    with pytest.raises(ValueError, match=argnone):
        _ = fitting.chi_sq(*args)


@pytest.mark.parametrize(
    "flux",
    [
        np.ones((4,)),
        np.ones((6, 7)),
        np.ones((4, 6, 2)),
    ],
)
@pytest.mark.parametrize(
    "ivar",
    [
        np.ones((8,)),
        np.ones((2, 2)),
        np.ones((5, 3, 1)),
    ],
)
def test_chisq_input_mismatch_flux_and_ivar(flux, ivar):
    with pytest.raises(ValueError, match="flux and ivar"):
        _ = fitting.chi_sq(np.ones(1), np.ones(1), flux, ivar)


@pytest.mark.parametrize(
    "theta",
    [
        np.ones((2, 4)),
        np.ones((6, 6)),
    ],
)
@pytest.mark.parametrize(
    "design_matrix",
    [
        np.ones((3,)),
        np.ones(
            (2, 4)
        ),  # Should not combine OK with same flux shape above - is transposed in calculation
    ],
)
@pytest.mark.parametrize(
    "flux",
    [
        np.ones((3,)),
        np.ones((6, 4, 2)),
    ],
)
def test_chisq_input_bad_shapes(theta, design_matrix, flux):
    with pytest.raises(ValueError, match="inconsistent shapes"):
        _ = fitting.chi_sq(
            theta, design_matrix, flux, flux
        )  # Let shape of ivar == shape flux
