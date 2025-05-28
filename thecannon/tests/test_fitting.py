"""
Unit tests for `thecannon.fitting`.
"""

import pytest
from unittest import mock
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
        (
            bad_value
            if arg == "flux"
            else np.ones(
                10,
            )
        ),
        (
            bad_value
            if arg == "ivar"
            else np.ones(
                10,
            )
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
    "bad_reg",
    [-1.0, -0.0001, np.asarray([1.0]), np.asarray([-1, 1]), "1", "a", "stuff"],
)
def test__pixel_objective_function_fixed_scatter(bad_reg):
    with pytest.raises(ValueError, match="must be a positive, finite number"):
        _ = fitting._pixel_objective_function_fixed_scatter(
            None, None, None, None, bad_reg
        )


@pytest.mark.parametrize(
    "residuals_squared",
    [
        np.ones((3,)),
        np.ones((3, 3, 3)),
        np.ones((4, 4)),
    ],
)
@pytest.mark.parametrize(
    "ivar",
    [
        np.ones((4,)),
        np.ones((3, 3)),
        np.ones((4, 4, 4)),
        np.ones((20,)),
    ],
)
def test__scatter_objective_function_bad_shapes(residuals_squared, ivar):
    with pytest.raises(ValueError):
        _ = fitting._scatter_objective_function(1.0, residuals_squared, ivar)


@pytest.mark.parametrize("scatter", [0.0, 1.0, 2.5, 0.01, 100])
@pytest.mark.parametrize(
    "residuals_squared,ivar",
    [
        (np.ones(10), 2.0 * np.ones(10)),
        (np.zeros(100), 0.03 * np.ones(100)),
    ],
)
def test_scatter_objective_function(scatter, residuals_squared, ivar):
    scat = fitting._scatter_objective_function(scatter, residuals_squared, ivar)
    assert scat.shape == (), "_scatter_objective_function should return a number"


@pytest.mark.parametrize("bad_method", ["a", "not_a_method", 1, 1.0, None])
def test__remove_forbidden_op_kwds_bad_method(bad_method):
    with pytest.raises(ValueError, match=f"{bad_method}"):
        _ = fitting._remove_forbidden_op_kwds(bad_method, {})


@pytest.mark.parametrize("method", ["l_bfgs_b", "powell"])
@pytest.mark.parametrize(
    "forbidden_kw",
    [
        ["a", "the"],
        ["these", "are", "not", "keywords"],
    ],
)
def test__remove_forbidden_op_kwds(method, forbidden_kw):
    kwg = {k: None for k in fitting.FITTING_ALLOWED_OPTS[method]}
    for k in forbidden_kw:
        kwg[k] = None

    fitting._remove_forbidden_op_kwds(method, kwg)

    assert set(kwg.keys()) == set(
        fitting.FITTING_ALLOWED_OPTS[method]
    ), "Failed to remove all bad keys"


@pytest.mark.parametrize(
    "flux",
    [
        np.ones((6,)),
        np.ones((6, 6)),
    ],
)
@pytest.mark.parametrize(
    "ivar",
    [
        np.ones(5),
        np.ones((5, 5)),
    ],
)
@pytest.mark.parametrize(
    "design_matrix",
    [
        np.ones((7, 8)),
        np.ones((2, 11)),
    ],
)
def test_fit_pixel_fixed_scatter_bad_array_sizes(flux, ivar, design_matrix):
    with pytest.raises(ValueError, match="shape"):
        _ = fitting.fit_pixel_fixed_scatter(flux, ivar, None, design_matrix, None, None)


def _fake_pixel_obj(first_input, *args):
    return first_input


@pytest.mark.parametrize(
    "flux,ivar,design_matrix",
    [
        (
            np.ones(
                10,
            ),
            np.ones(
                10,
            ),
            np.ones((10, 5)),
        )
    ],
)
@pytest.mark.parametrize("regularization", [0.1, 1.0, 10.0])
class TestFitPixelFixedScatterSundries:

    @pytest.mark.parametrize(
        "initial_theta",
        [
            [(np.zeros(7), "Bad guess")],
            [(np.zeros(5), "First one is good"), (np.zeros(9), "Second one is bad")],
        ],
    )
    def test_fit_pixel_fixed_scatter_secondary_bad_theta(
        self, initial_theta, design_matrix, flux, ivar, regularization
    ):
        with pytest.raises(ValueError):
            _ = fitting.fit_pixel_fixed_scatter(
                flux, ivar, initial_theta, design_matrix, regularization, None
            )

    @pytest.mark.parametrize(
        "initial_thetas",
        [
            [(0.1, "Pick me"), (np.nan, "Don't pick me"), (10.0, "Don't pick me")],
            [(np.nan, "Don't pick me"), (0.01, "Pick me"), (10.0, "Don't pick me")],
            [
                (100.0, "Don't pick me"),
                (0.01, "Pick me"),
            ],
        ],
    )
    def test__select_theta(
        self, initial_thetas, design_matrix, flux, ivar, regularization
    ):
        # Find what value we should get
        for theta, msg in initial_thetas:
            if msg == "Pick me":
                target_theta = theta
                break

        with mock.patch(
            "thecannon.fitting._pixel_objective_function_fixed_scatter",
            wraps=_fake_pixel_obj,
        ) as mock_obj:
            t, s = fitting._select_theta(
                flux, ivar, initial_thetas, design_matrix, regularization
            )
            assert t == target_theta
            assert s == "Pick me"

    @pytest.mark.parametrize(
        "bad_method",
        [
            "a",
            "stuff",
            1.0,
            3,
            True,
        ],
    )
    def test_fit_pixel_fixed_scatter_bad_method(
        self, bad_method, design_matrix, flux, ivar, regularization
    ):
        with pytest.raises(ValueError, match="unknown optimization"):
            _ = fitting.fit_pixel_fixed_scatter(
                flux, ivar, None, design_matrix, 1.0, None, op_method=bad_method
            )
