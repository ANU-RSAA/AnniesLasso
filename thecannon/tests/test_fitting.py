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
        np.zeros((24, ))
    ],
)
@pytest.mark.parametrize(
    "ivar",
    [
        np.ones((8,)),
        np.zeros((16, ))
    ],
)
def test_chisq_input_mismatch_flux_and_ivar(flux, ivar):
    with pytest.raises(ValueError, match="flux and ivar"):
        _ = fitting.chi_sq(np.ones(1), np.ones((1, 1)), flux, ivar)

@pytest.mark.parametrize("bad_value", [
    np.ones((4, 2)),
    np.zeros((9, 6)),
])
@pytest.mark.parametrize("arg", ["flux", "ivar"])
def test_chisq_2d_flux_or_ivar(bad_value, arg):
    with pytest.raises(ValueError, match="one-dimensional"):
        _ = fitting.chi_sq(
            np.ones((5, )),
            np.ones((10, 5)),
            bad_value if arg == "flux" else np.ones(10, ),
            bad_value if arg == "ivar" else np.ones(10, ),
        )


# Every number is different to ensure coordinate mis-match somewhere
@pytest.mark.parametrize(
    "theta",
    [
        np.ones((2, )),
        np.ones((6, )),
    ],
)
@pytest.mark.parametrize(
    "design_matrix",
    [
        np.ones((3, 11)),
        np.ones(
            (2, 4)
        ),  # Should not combine OK with same flux shape above - is transposed in calculation
    ],
)
@pytest.mark.parametrize(
    "flux",
    [
        np.ones((3,)),
    ],
)
def test_chisq_input_bad_shapes(theta, design_matrix, flux):
    with pytest.raises(ValueError, match="inconsistent shapes"):
        _ = fitting.chi_sq(
            theta, design_matrix, flux, flux
        )  # Let shape of ivar == shape flux


@pytest.mark.parametrize("design_matrix", [
    np.ones((10, )),
    np.zeros((10, 10, 10, ))
])
def test_chisq_input_bad_design_matrix_shape(design_matrix):
    with pytest.raises(ValueError, match="design_matrix must be"):
        _ = fitting.chi_sq(np.ones(10), design_matrix, np.ones(10), np.ones(10))

@pytest.mark.parametrize("P", [10, 100, 1000])
@pytest.mark.parametrize("S", [5, 50, 500])
@pytest.mark.parametrize("T", [3, 30, 90])
def test_chisq_return_formats(P, S, T):
    theta = np.ones(T)
    design_matrix = np.ones((S, T)) * 2
    flux = np.ones(S) * 5
    ivar = np.ones(S) * 0.1

    # Base case - should return a chi_sq, gradient tuple
    ch = fitting.chi_sq(theta, design_matrix, flux, ivar)
    with pytest.raises(AttributeError):
        _ = ch.shape  # Will fail if c is an array
    assert len(ch) == 2, "Did not return a 2-tuple as expected"
    assert ch[1].shape == (T, ), "Expected a gradient/Jacobian with shape (T, )"
