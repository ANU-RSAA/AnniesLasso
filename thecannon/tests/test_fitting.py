"""
Unit tests for `thecannon.fitting`.
"""

import pytest
import numpy as np
from thecannon import fitting


@pytest.mark.parametrize("argnone", ["theta", "design_matrix", "flux", "ivar"])
@pytest.mark.parametrize("f", ["chi_sq", "_pixel_objective_function_fixed_scatter"])
def test_chisq_input_none(argnone, f):
    dummy_array = np.ones((1,))
    args = [
        None if k == argnone else dummy_array
        for k in ["theta", "design_matrix", "flux", "ivar"]
    ]
    if f == "_pixel_objective_function_fixed_scatter":
        args.append(1.0)
    with pytest.raises(ValueError, match=argnone):
        _ = getattr(fitting, f)(*args)


@pytest.mark.parametrize(
    "flux",
    [np.ones((4,)), np.zeros((24,))],
)
@pytest.mark.parametrize(
    "ivar",
    [np.ones((8,)), np.zeros((16,))],
)
@pytest.mark.parametrize("f", ["chi_sq", "_pixel_objective_function_fixed_scatter"])
def test_chisq_input_mismatch_flux_and_ivar(flux, ivar, f):
    with pytest.raises(ValueError, match="flux and ivar"):
        if f == "chi_sq":
            _ = fitting.chi_sq(np.ones(1), np.ones((1, 1)), flux, ivar)
        else:
            _ = fitting._pixel_objective_function_fixed_scatter(
                np.ones(1), np.ones((1, 1)), flux, ivar, 1.0
            )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.ones((4, 2)),
        np.zeros((9, 6)),
    ],
)
@pytest.mark.parametrize("arg", ["flux", "ivar"])
@pytest.mark.parametrize("f", ["chi_sq", "_pixel_objective_function_fixed_scatter"])
def test_chisq_2d_flux_or_ivar(bad_value, arg, f):
    args = [
        np.ones((5,)),
        np.ones((10, 5)),
        bad_value
        if arg == "flux"
        else np.ones(
            10,
        ),
        bad_value
        if arg == "ivar"
        else np.ones(
            10,
        ),
    ]
    if f == "_pixel_objective_function_fixed_scatter":
        args.append(1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        _ = getattr(fitting, f)(*args)


# Every number is different to ensure coordinate mis-match somewhere
@pytest.mark.parametrize(
    "theta",
    [
        np.ones((2,)),
        np.ones((6,)),
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
@pytest.mark.parametrize("f", ["chi_sq", "_pixel_objective_function_fixed_scatter"])
def test_chisq_input_bad_shapes(theta, design_matrix, flux, f):
    args = [theta, design_matrix, flux, flux]  # Let shape of ivar == shape flux
    if f == "_pixel_objective_function_fixed_scatter":
        args.append(1.0)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        _ = getattr(fitting, f)(*args)


@pytest.mark.parametrize(
    "design_matrix",
    [
        np.ones((10,)),
        np.zeros(
            (
                10,
                10,
                10,
            )
        ),
    ],
)
@pytest.mark.parametrize("f", ["chi_sq", "_pixel_objective_function_fixed_scatter"])
def test_chisq_input_bad_design_matrix_shape(design_matrix, f):
    args = [np.ones(10), design_matrix, np.ones(10), np.ones(10)]
    if f == "_pixel_objective_function_fixed_scatter":
        args.append(1.0)
    with pytest.raises(ValueError, match="design_matrix must be"):
        _ = getattr(fitting, f)(*args)


@pytest.mark.parametrize("P", [10, 100, 1000])
@pytest.mark.parametrize("S", [5, 50, 500])
@pytest.mark.parametrize("T", [3, 30, 90])
@pytest.mark.parametrize(
    "chisq_expected",
    [
        {
            # "S,T": <expected value>
            "5,3": 0.5,
            "50,3": 5.0,
            "500,3": 50.0,
            "5,30": 1512.5,
            "50,30": 15125.0,
            "500,30": 151250.0,
            "5,90": 15312.5,
            "50,90": 153125.0,
            "500,90": 1531250.0,
        }
    ],
)
@pytest.mark.parametrize(
    "Jacob_expected",
    [
        {
            "5,3": 2.0,
            "50,3": 20.0,
            "500,3": 200.0,
            "5,30": 110.0,
            "50,30": 1100.0,
            "500,30": 11000.0,
            "5,90": 350.0,
            "50,90": 3500.0,
            "500,90": 35000.0,
        }
    ],
)
class TestHelperFunctionReturns:
    def test_chisq_return_formats(self, P, S, T, chisq_expected, Jacob_expected):
        theta = np.ones(T)
        design_matrix = np.ones((S, T)) * 2
        flux = np.ones(S) * 5
        ivar = np.ones(S) * 0.1

        # Base case - should return a chi_sq, gradient tuple
        ch = fitting.chi_sq(theta, design_matrix, flux, ivar)
        with pytest.raises(AttributeError):
            _ = ch.shape  # Will fail if c is an array
        assert len(ch) == 2, "Did not return a 2-tuple as expected"
        assert ch[1].shape == (T,), "Expected a gradient/Jacobian with shape (T, )"
        assert np.all(
            ch[1] == pytest.approx(Jacob_expected[f"{S},{T}"])
        ), f"Did not get expected gradient/Jacobian"

        # Now, dont ask for the gradient
        ch2 = fitting.chi_sq(theta, design_matrix, flux, ivar, gradient=False)
        assert (
            ch2.shape == ()
        ), "Bad return without gradient"  # Will fail if c is an array
        assert not isinstance(
            ch2, tuple
        ), "Setting gradient=False still returned a tuple"

        assert ch[0] == pytest.approx(
            ch2
        ), "Returned different values from different gradient kwarg value"
        assert ch2 == pytest.approx(
            chisq_expected[f"{S},{T}"]
        ), f"Wrong chi_sq value returned ({ch2} != {chisq_expected[f'{S},{T}']})"

    @pytest.mark.parametrize("reg", [1.0, 2.5, 10.0, 0.01, 0.5])
    def test__pixel_objective_function_fixed_scatter(
        self, reg, P, S, T, chisq_expected, Jacob_expected
    ):
        theta = np.ones(T)
        design_matrix = np.ones((S, T)) * 2
        flux = np.ones(S) * 5
        ivar = np.ones(S) * 0.1

        # Base case - should return a two-tuple
        fg = fitting._pixel_objective_function_fixed_scatter(
            theta, design_matrix, flux, ivar, reg
        )
        with pytest.raises(AttributeError):
            _ = fg.shape  # Will fail if fg is an array
        assert len(fg) == 2 and isinstance(
            fg, tuple
        ), "Did not return a 2-tuple as expected"
        (f1, g) = fg
        assert g.shape == (T,), "Expected a gradient/Jacobian with shape (T, )"
        assert np.all(
            Jacob_expected[f"{S},{T}"] + reg * fitting.L1Norm_variation(theta)[1]
            == pytest.approx(g)
        ), "Something has changed in gradient/Jacobian calculation"

        # Now, without the gradient
        f = fitting._pixel_objective_function_fixed_scatter(
            theta, design_matrix, flux, ivar, reg, gradient=False
        )
        assert f.shape == (), "Bad return shape"
        assert not isinstance(f, tuple), "Setting gradient=False still returned a tuple"
        assert f1 == pytest.approx(
            f
        ), "Returned different values based on gradient kwarg"
        assert (
            pytest.approx(f)
            == chisq_expected[f"{S},{T}"] + reg * fitting.L1Norm_variation(theta)[0]
        ), "Something has changed in objective func calculation"


@pytest.mark.parametrize(
    "bad_theta",
    [
        np.ones((3, 3)),
        np.ones(1),
    ],
)
def test_L1Norm_variation_bad_input(bad_theta):
    with pytest.raises(ValueError, match="theta must"):
        _ = fitting.L1Norm_variation(bad_theta)


@pytest.mark.parametrize(
    "theta",
    [
        np.ones(100) * -1,
        np.zeros(50),
        np.asarray(range(1000)),
    ],
)
def test_L1Norm_variation(theta):
    L1 = fitting.L1Norm_variation(theta)
    assert isinstance(L1, tuple) and len(L1) == 2, "Did not return a 2-tuple"
    assert L1[0] == np.sum(np.abs(theta[1:])), "Calculation of L1 norm has changed"
    assert np.all(
        L1[1] == np.hstack([0.0, np.sign(theta[1:])])
    ), "Calculation of L1 norm direction/gradient has changed"


@pytest.mark.parametrize(
    "bad_reg", [-1.0, -0.0001, np.asarray([1.0]), np.asarray([-1, 1]), "1", "a", "stuff"]
)
def test__pixel_objective_function_fixed_scatter(bad_reg):
    with pytest.raises(ValueError, match="must be a positive, finite number"):
        _ = fitting._pixel_objective_function_fixed_scatter(
            None, None, None, None, bad_reg
        )

@pytest.mark.parametrize("residuals_squared", [
    np.ones((3, )),
    np.ones((3, 3, 3)),
    np.ones((4, 4)),
])
@pytest.mark.parametrize("ivar", [
    np.ones((4, )),
    np.ones((3, 3)),
    np.ones((4, 4, 4)),
    np.ones((20, )),
])
def test__scatter_objective_function_bad_shapes(residuals_squared, ivar):
    with pytest.raises(ValueError):
        _ = fitting._scatter_objective_function(1.0, residuals_squared, ivar)

@pytest.mark.parametrize("scatter", [
    0.0,
    1.0, 
    2.5,
    0.01,
    100
])
@pytest.mark.parametrize("residuals_squared,ivar", [
    (np.ones(10), 2.0 * np.ones(10)),
    (np.zeros(100), 0.03 * np.ones(100)),
])
def test_scatter_objective_function(scatter, residuals_squared, ivar):
    scat = fitting._scatter_objective_function(scatter, residuals_squared, ivar)
    assert scat.shape == (), "_scatter_objective_function should return a number"
