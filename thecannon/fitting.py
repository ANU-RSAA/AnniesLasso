#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fitting functions for use in The Cannon.
"""

__all__ = [
    "fit_spectrum",
    "fit_pixel_fixed_scatter",
    "fit_theta_by_linalg",
    "chi_sq",
    "L1Norm_variation",
]

import logging
import numpy as np
import scipy.optimize as op
from time import time

logger = logging.getLogger(__name__)

FITTING_COMMON_KEYS = (
    "x0",
    "args",
    "method",
    "jac",
    "hess",
    "hessp",
    "bounds",
    "constraints",
    "tol",
    "callback",
)

FITTING_ALLOWED_OPTS = dict(
    l_bfgs_b=(
        "maxcor",
        "ftol",
        "gtol",
        "eps",
        "maxfun",
        "maxiter",
        "maxls",
        "finite_diff_rel_step",
    ),
    powell=(
        "disp",
        "xtol",
        "ftol",
        "maxiter",
        "maxfev",
        "direc",
        "return_all",
    ),
)

PIXEL_FITTING_METHODS = (
    "l_bfgs_b",
    "powell",
)


def fit_spectrum(
    flux,
    ivar,
    initial_labels,
    vectorizer,
    theta,
    s2,
    fiducials,
    scales,
    dispersion=None,
    use_derivatives=True,
    op_kwds=None,
):
    """
    Fit a single spectrum by least-squared fitting.

    As this function fits a single full spectrum, all arrays mentioned are of
    shape ``(P, )``, where ``P`` is the number of pixels in the spectrum.

    Parameters
    ----------
    flux : 1D array
        The normalized flux values.
    ivar : 1D array
        The inverse variance array for the normalized fluxes.
    initial_labels : 1D array
        The point(s) to initialize optimization from.
    vectorizer : :py:class:`BaseVectorizer` instance
        The vectorizer to use when fitting the data.
    theta : 2D array
        The theta coefficients (spectral derivatives) of the trained model.
        The shape of this array is ``(P, T)``, where ``T`` is the number of terms
        in the model (including the regularization term).
    s2 : 1D array
        The pixel scatter (:math:`s^2`) array for each pixel.
    dispersion : optional
        The dispersion (e.g., wavelength) points for the normalized fluxes.
    use_derivatives : Boolean or callable, optional
        ``True`` indicates to use analytic derivatives provided by
        the vectorizer, ``None`` to calculate on the fly, or a callable
        function to calculate your own derivatives.
    op_kwds : dict, optional
        Optimization keywords that get passed to :py:meth:`scipy.optimize.leastsq`.

    Returns
    -------
    3-tuple
        A three-length tuple containing: the optimized labels, the covariance
        matrix, and metadata associated with the optimization.
    """

    adjusted_ivar = ivar / (1.0 + ivar * s2)

    # Exclude non-finite points (e.g., points with zero inverse variance
    # or non-finite flux values, but the latter shouldn't exist anyway).
    use = np.isfinite(flux * adjusted_ivar) * (adjusted_ivar > 0)
    L = len(vectorizer.label_names)

    if not np.any(use):
        logger.warning("No information in spectrum!")
        return (
            np.nan * np.ones(L),
            np.nan * np.ones((L, L)),
            {"fail_message": "Pixels contained no information"},
        )

    # Splice the arrays we will use most.
    flux = flux[use]
    weights = np.sqrt(adjusted_ivar[use])  # --> 1.0 / sigma
    use_theta = theta[use]

    initial_labels = np.atleast_2d(initial_labels)

    # Check the vectorizer whether it has a derivative built in.
    if use_derivatives not in (None, False):
        try:
            vectorizer.get_label_vector_derivative(initial_labels[0])

        except NotImplementedError:
            Dfun = None
            logger.warning(
                "No label vector derivatives available in {}!".format(vectorizer)
            )

        except:
            logger.exception(
                "Exception raised when trying to calculate the "
                "label vector derivative at the fiducial values:"
            )
            raise

        else:
            # Use the label vector derivative.
            Dfun = (
                lambda parameters: weights
                * np.dot(
                    use_theta, vectorizer.get_label_vector_derivative(parameters)
                ).T
            )

    else:
        Dfun = None

    def func(parameters):
        return np.dot(use_theta, vectorizer(parameters))[:, 0]

    def residuals(parameters):
        return weights * (func(parameters) - flux)

    kwds = {
        "func": residuals,
        "Dfun": Dfun,
        "col_deriv": True,
        # These get passed through to leastsq:
        "ftol": 7.0 / 3 - 4.0 / 3 - 1,  # Machine precision.
        "xtol": 7.0 / 3 - 4.0 / 3 - 1,  # Machine precision.
        "gtol": 0.0,
        "maxfev": 100000,  # MAGIC
        "epsfcn": None,
        "factor": 1.0,
    }

    # Only update the keywords with things that op.curve_fit/op.leastsq expects.
    if op_kwds is not None:
        for key in set(op_kwds).intersection(kwds):
            kwds[key] = op_kwds[key]

    results = []
    for x0 in initial_labels:
        try:
            op_labels, cov, meta, mesg, ier = op.leastsq(
                x0=(x0 - fiducials) / scales, full_output=True, **kwds
            )

        except RuntimeError:
            logger.exception("Exception in fitting from {}".format(x0))
            continue

        meta.update(dict(x0=x0, chi_sq=np.sum(meta["fvec"] ** 2), ier=ier, mesg=mesg))
        results.append((op_labels, cov, meta))

    if len(results) == 0:
        logger.warning("No results found!")
        return (np.nan * np.ones(L), None, dict(fail_message="No results found"))

    best_result_index = np.nanargmin([m["chi_sq"] for (o, c, m) in results])
    op_labels, cov, meta = results[best_result_index]

    # De-scale the optimized labels.
    meta["model_flux"] = func(op_labels)
    op_labels = op_labels * scales + fiducials

    if np.allclose(op_labels, meta["x0"]):
        logger.warning(
            "Discarding optimized result because it is exactly the same as the "
            "initial value!"
        )

        # We are in dire straits. We should not trust the result.
        op_labels *= np.nan
        meta["fail_message"] = "Optimized result same as initial value."

    if cov is None:
        cov = np.ones((len(op_labels), len(op_labels)))

    if not np.any(np.isfinite(cov)):
        logger.warning("Non-finite covariance matrix returned!")

    # Save additional information.
    meta.update(
        {
            "method": "leastsq",
            "label_names": vectorizer.label_names,
            "best_result_index": best_result_index,
            "derivatives_used": Dfun is not None,
            "snr": np.nanmedian(flux * weights),
            "r_chi_sq": meta["chi_sq"] / (use.sum() - L - 1),
        }
    )
    for key in ("ftol", "xtol", "gtol", "maxfev", "factor", "epsfcn"):
        meta[key] = kwds[key]

    return (op_labels, cov, meta)


def fit_theta_by_linalg(flux, ivar, s2, design_matrix):
    """
    Fit theta coefficients to a set of normalized fluxes for a single pixel.

    Parameters
    ----------
    flux: 1D array
        The normalized fluxes for a single pixel (across many stars), shape ``(S, )``.
    ivar: 1D array
        The inverse variance of the normalized flux values for a single pixel
        across many stars.
    s2: float
        The noise residual (squared scatter term) to adopt in the pixel.
    design_matrix: 2D array
        The model design matrix.

    Returns
    -------
    theta, ARCiAinv
        The label vector coefficients for the pixel, and the inverse variance
        matrix.
    """

    adjusted_ivar = ivar / (1.0 + ivar * s2)
    CiA = design_matrix * np.tile(adjusted_ivar, (design_matrix.shape[1], 1)).T
    try:
        ATCiAinv = np.linalg.inv(np.dot(design_matrix.T, CiA))
    except np.linalg.linalg.LinAlgError:
        N = design_matrix.shape[1]
        return (np.hstack([1, np.zeros(N - 1)]), np.inf * np.eye(N))

    ATY = np.dot(design_matrix.T, flux * adjusted_ivar)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv)


# TODO: This logic should probably go somewhere else.


def chi_sq(theta, design_matrix, flux, ivar, axis=None, gradient=True):
    """
    Calculate the chi-squared difference between the spectral model and flux, for a single
    pixel over multiple stars.

    Assume a model has ``S`` stars, each of ``P`` pixels, and a model with `T` terms (including the
    regularization term).

    Parameters
    ----------
    theta : 1D array
        The theta coefficients for this pixel (shape ``(T, )``).
    design_matrix : 2D array
        The model design matrix (shape ``(S, T)``.)
    flux : 1D array
        The normalized flux values for this pixel (shape ``(S, )``).
    ivar : 1D array
        The inverse variances of the normalized flux values for this pixel (shape ``(S, )``).
    axi : int, optional
        The axis to sum the chi-squared values across.
    gradient : bool, optional
        Return the chi-squared value and its derivatives (Jacobian).

    Returns
    -------
    residuals, Jacobian
        The chi-squared difference between the spectral model and flux, and
        optionally, the Jacobian.
    """
    # Input checking
    if theta is None:
        raise ValueError("theta cannot be None")
    if design_matrix is None:
        raise ValueError("design_matrix cannot be None")
    if flux is None:
        raise ValueError("flux cannot be None")
    if ivar is None:
        raise ValueError("ivar cannot be None")
    if len(flux.shape) > 1 or len(ivar.shape) > 1 or len(theta.shape) > 1:
        raise ValueError(
            "flux, ivar and theta can only be one-dimensional here (for a given pixel)"
        )
    if len(design_matrix.shape) != 2:
        raise ValueError("design_matrix must be a 2D array, of shape (S, T)")
    if flux.shape != ivar.shape:
        raise ValueError("flux and ivar have inconsistent shapes")

    try:
        residuals = np.dot(theta, design_matrix.T) - flux
    except ValueError as e:
        raise ValueError(
            f"inconsistent shapes between theta {theta.shape}, design_matrix.T {design_matrix.T.shape} and flux {flux.shape}"
        ) from e

    ivar_residuals = ivar * residuals
    f = np.sum(ivar_residuals * residuals, axis=axis)
    if not gradient:
        return f

    # import pdb; pdb.set_trace()
    g = 2.0 * np.dot(design_matrix.T, ivar_residuals)
    return (f, g)


def L1Norm_variation(theta):
    """
    Return the L1 norm of theta (except the first entry) and its derivative.

    Parameters
    ----------
    theta: 1D array
        An array of finite values, shape ``(P, )``.

    Returns
    -------
    (1D array, 1D array)
        A two-length tuple containing: the L1 norm of theta (except the first
        entry), and the derivative of the L1 norm of theta.
    """
    if len(theta.shape) > 1:
        raise ValueError("theta must be single-dimensional")
    if theta.shape == (1,):
        raise ValueError("theta must have more than one element")

    return (np.sum(np.abs(theta[1:])), np.hstack([0.0, np.sign(theta[1:])]))


def _pixel_objective_function_fixed_scatter(
    theta, design_matrix, flux, ivar, regularization, gradient=False
):
    """
    The objective function for a single regularized pixel with fixed scatter.

    Parameters
    ----------
    theta:
        The spectral coefficients.
    design_matrix: 2D array
        The design matrix for the model.
    flux: 1D array
        The normalized flux values for a single pixel across many stars.
    ivar: 1D array
        The adjusted inverse variance of the normalized flux values for a single
        pixel across many stars. This adjusted inverse variance array should
        already have the scatter included.
    regularization: float
        The regularization term to scale the L1 norm of theta with.
    gradient: bool, optional
        Also return the analytic derivative of the objective function.
    """
    # No need to input check theta, design_matrix, flux, ivar - chi_sq will do that
    try:  # Check if finite, positive, and a single value
        assert (
            np.isfinite(regularization)
            and np.asarray(regularization).shape == ()
            and regularization >= 0.0
        )
    except (AssertionError, TypeError, ValueError):
        raise ValueError(
            f"regularization ({regularization}) must be a positive, finite number"
        )

    if gradient:
        csq, d_csq = chi_sq(theta, design_matrix, flux, ivar, gradient=True)
        L1, d_L1 = L1Norm_variation(theta)

        f = csq + regularization * L1
        g = d_csq + regularization * d_L1

        return (f, g)

    else:
        csq = chi_sq(theta, design_matrix, flux, ivar, gradient=False)
        L1, d_L1 = L1Norm_variation(theta)

        return csq + regularization * L1


def _scatter_objective_function(scatter, residuals_squared, ivar):
    # Input checking
    try:
        assert len(ivar.shape) == 1
        assert ivar.shape == residuals_squared.shape
    except AssertionError:
        raise ValueError(
            f"ivar shape {ivar.shape} does not match residuals_squared shape {residuals_squared.shape}"
        )
    adjusted_ivar = ivar / (1.0 + ivar * scatter**2)
    chi_sq = residuals_squared * adjusted_ivar
    return (np.median(chi_sq) - 1.0) ** 2

def _pixel_objective_function_fixed_scatter_jac(
    theta, design_matrix, flux, ivar, regularization
):
    return _pixel_objective_function_fixed_scatter(theta, design_matrix, flux, ivar, regularization, gradient=True)[1]


def _remove_forbidden_op_kwds(op_method, op_kwds):
    """
    Remove forbidden optimization keywords.

    Parameters
    ----------
    op_method: str
        The optimization algorithm to use.
    op_kwds: dict
        Optimization keywords.

    Returns
    -------
    None
        `None`. The dictionary of `op_kwds` will be updated.
    """
    try:
        forbidden_keys = set(op_kwds).difference(FITTING_ALLOWED_OPTS[op_method] + FITTING_COMMON_KEYS)
    except KeyError:
        raise ValueError(f"Unknown op_method {op_method}")
    if forbidden_keys:
        logger.warning(
            "Ignoring forbidden optimization keywords for {}: {}".format(
                op_method, ", ".join(forbidden_keys)
            )
        )
        for key in forbidden_keys:
            del op_kwds[key]

    return None


def _select_theta(flux, ivar, initial_thetas, design_matrix, regularization):
    """
    Select the best theta to use in `fit_pixel_fixed_scatter`.

    Parameters
    ----------
    initial_thetas: list of 2-tuples
        A list of initial theta values to start from, and their source. For
        example: `[(theta_0, "guess"), (theta_1, "old_theta")]`
    design_matrix: 2D array
        The model design matrix (shape `(S, T)`.)
    regularization: float
        The regularization term to scale the L1 norm of theta with.

    Returns
    -------
    (initial_theta: float, initial_theta_source: str)
        The best initial theta values from the input, plus its source.
    """

    feval = []
    for initial_theta, initial_theta_source in initial_thetas:
        feval.append(
            _pixel_objective_function_fixed_scatter(
                initial_theta, design_matrix, flux, ivar, regularization, False
            )
        )

    initial_theta, initial_theta_source = initial_thetas[np.nanargmin(feval)]

    return (initial_theta, initial_theta_source)


def fit_pixel_fixed_scatter(
    flux, ivar, initial_thetas, design_matrix, regularization, censoring_mask, **kwargs
):
    """
    Fit theta coefficients and noise residual for a single pixel, using
    an initially fixed scatter value.

    Parameters
    ----------
    flux: 1D array
        The normalized flux values, shape ``(S, )``.
    ivar: 1D array
        The inverse variance array for the normalized fluxes.
    initial_thetas: 1D array
        A list of initial theta values to start from, and their source. For
        example: ``[(theta_0, "guess"), (theta_1, "old_theta")]``
    design_matrix: 2D array
        The model design matrix.
    regularization: float
        The regularization strength to apply during optimization (Lambda).
    censoring_mask: `Censor` object
        A per-label censoring mask for each pixel.
    op_method: str, optional
        The optimization method to use. Valid options are: ``"l_bfgs_b"``, ``"powell"``.
    op_kwds: dict, optional
        A dictionary of arguments that will be provided to the optimizer.

    Returns
    -------
    (2D array, 2D array, dict)
        The optimized theta coefficients, the noise residual ``s2``, and
        metadata related to the optimization process.
    """
    # Input checking
    if len(flux.shape) != 1 or ivar.shape != flux.shape:
        raise ValueError("flux and ivar must be 1D and of the same shape")
    if design_matrix.shape[0] != flux.shape[0]:
        raise ValueError("design_matrix first axis shape must match flux/ivar shape")
    # Only need to check these inputs up to this point
    if np.sum(ivar) < 1.0 * ivar.size:  # MAGIC
        metadata = dict(message="No pixel information.", op_time=0.0)
        fiducial = np.hstack([1.0, np.zeros(design_matrix.shape[1] - 1)])
        return (fiducial, np.inf, metadata)  # MAGIC

    # If we get past here, need to input check the rest
    # Note that we may be able to do some input checking by catching errors further below,
    # if not too far down
    # Allow either l_bfgs_b or powell
    default_op_method = "l_bfgs_b"
    op_method = kwargs.get("op_method", default_op_method) or default_op_method
    op_method = str(op_method).lower()
    if op_method not in PIXEL_FITTING_METHODS:
        raise ValueError(
            "unknown optimization method '{}' -- "
            "{} are available".format(op_method, ",".join(PIXEL_FITTING_METHODS))
        )
    op_strict = kwargs.get("op_strict", True)

    # Determine if any theta coefficients will be censored.
    censored_theta = ~np.any(np.isfinite(design_matrix), axis=0)
    # Make the design matrix safe to use.
    design_matrix[:, censored_theta] = 0

    # These calls will check the initial_thetas, design_matrix, and regularization values
    initial_theta, initial_theta_source = _select_theta(
        flux, ivar, initial_thetas, design_matrix, regularization
    )

    base_op_kwds = dict(
        x0=initial_theta,
        args=(design_matrix, flux, ivar, regularization),
        maxfun=np.inf,
        maxiter=np.inf,
    )

    theta_0 = kwargs.get("__theta_0", None)
    if theta_0 is not None:
        logger.warning("FIXING theta_0. HIGHLY EXPERIMENTAL.")

        # Subtract from flux.
        # Set design matrix entry to zero.
        # Update to theta later on.
        try:
            new_flux = flux - theta_0
            new_design_matrix = np.copy(design_matrix)
            new_design_matrix[:, 0] = 0.0
        except ValueError as e:
            raise ValueError("theta_0 shape incompatible with flux/design_matrix shape")

        base_op_kwds["args"] = (new_design_matrix, new_flux, ivar, regularization)

    if any(censored_theta):
        # If the initial_theta is the same size as the censored_mask, but different
        # to the design_matrix, then we need to censor the initial theta so that we
        # don't bother solving for those parameters.
        base_op_kwds["x0"] = np.array(base_op_kwds["x0"])[~censored_theta]
        base_op_kwds["args"] = (
            design_matrix[:, ~censored_theta],
            flux,
            ivar,
            regularization,
        )

    t_init = time()

    while True:
        if op_method == "l_bfgs_b":
            op_kwds = dict()
            op_kwds.update(base_op_kwds)
            # FIXME shift to constants
            op_kwds.update(maxcor=design_matrix.shape[1], maxls=20, ftol=10.0 * np.finfo(float).eps, gtol=1e-6)
            op_kwds.update((kwargs.get("op_kwds", {}) or {}))

            # If op_bounds are given and we are censoring some theta terms, then we
            # will need to adjust which op_bounds we provide.
            if "bounds" in op_kwds and any(censored_theta):
                op_kwds["bounds"] = [
                    b
                    for b, is_censored in zip(op_kwds["bounds"], censored_theta)
                    if not is_censored
                ]

            # Just-in-time to remove forbidden keywords.
            _remove_forbidden_op_kwds(op_method, op_kwds)

            # op_params, fopt, metadata 
            op_return = op.minimize(
                _pixel_objective_function_fixed_scatter,
                jac=_pixel_objective_function_fixed_scatter_jac,
                method="L-BFGS-B",
                # fprime=None,
                # approx_grad=None,
                options={k:v for k,v in op_kwds.items() if k in FITTING_ALLOWED_OPTS[op_method]},
                **{k:v for k,v in op_kwds.items() if k in FITTING_COMMON_KEYS},
            )
            op_params = op_return.x
            fopt = op_return.fun
            metadata = {
                "warnflag": op_return.status,
                "funcalls": op_return.nfev,
                "nit": op_return.nit,
                "message": op_return.message,
            }
            metadata.update(dict(fopt=fopt))

            warnflag = metadata.get("warnflag", -1)
            if warnflag > 0:
                reason = (
                    "too many function evaluations or too many iterations"
                    if warnflag == 1
                    else metadata["message"]
                )
                logger.warning("Optimization warning (l_bfgs_b): {}".format(reason))

                if op_strict:
                    # Do optimization again.
                    op_method = "powell"
                    base_op_kwds.update(x0=op_params)
                else:
                    break

            else:
                break

        elif op_method == "powell":
            op_kwds = dict()
            op_kwds.update(base_op_kwds)
            op_kwds.update(xtol=1e-6, ftol=1e-6)
            del(op_kwds["maxfun"])
            op_kwds.update((kwargs.get("op_kwds", {}) or {}))

            # Set 'False' in args so that we don't return the gradient,
            # because fmin doesn't want it.
            args = list(op_kwds["args"])
            args.append(False)
            op_kwds["args"] = tuple(args)

            t_init = time()

            # Just-in-time to remove forbidden keywords.
            _remove_forbidden_op_kwds(op_method, op_kwds)

            op_return = op.minimize(
                _pixel_objective_function_fixed_scatter,
                jac=_pixel_objective_function_fixed_scatter_jac,
                method="Powell",
                options={k:v for k,v in op_kwds.items() if k in FITTING_ALLOWED_OPTS[op_method]},
                **{k:v for k,v in op_kwds.items() if k in FITTING_COMMON_KEYS},
            )
            op_params = op_return.x
            fopt = op_return.fun
            warnflag = op_return.status
            n_iter = op_return.nit
            n_funcs = op_return.nfev
            direc = op_return.direc

            metadata = dict(
                fopt=fopt,
                direc=direc,
                n_iter=n_iter,
                n_funcs=n_funcs,
                warnflag=warnflag,
            )
            break

        else:
            raise ValueError(
                "unknown optimization method '{}' -- "
                "powell or l_bfgs_b are available".format(op_method)
            )

    # Additional metadata common to both optimizers.
    metadata.update(
        dict(
            op_method=op_method,
            op_time=time() - t_init,
            initial_theta=initial_theta,
            initial_theta_source=initial_theta_source,
        )
    )

    # De-censor the optimized parameters.
    if any(censored_theta):
        theta = np.zeros(censored_theta.size)
        theta[~censored_theta] = op_params

    else:
        theta = op_params

    if theta_0 is not None:
        theta[0] = theta_0

    # Fit the scatter.
    op_fmin_kwds = dict(disp=False, maxiter=np.inf, maxfun=np.inf)
    op_fmin_kwds.update(xtol=op_kwds.get("xtol", 1e-8), ftol=op_kwds.get("ftol", 1e-8))

    residuals_squared = (flux - np.dot(theta, design_matrix.T)) ** 2
    scatter = op.fmin(
        _scatter_objective_function, 0.0, args=(residuals_squared, ivar), disp=False
    )

    return (theta, scatter**2, metadata)
