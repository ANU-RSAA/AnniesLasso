#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Cannon.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ["CannonModel"]

import logging
import multiprocessing as mp
import numpy as np
import os
import pickle
from datetime import datetime
from functools import wraps
from sys import version_info
from scipy.spatial import Delaunay

from .vectorizer.base import BaseVectorizer
from . import censoring, fitting, utils, vectorizer as vectorizer_module, __version__


logger = logging.getLogger(__name__)


def _compare_none_or_arrays(
    first, second, rtol=1e-05, atol=1e-08, equal_nan=False, allow_one_none=False
):
    """
    Compare arguments for equality, where they are expected to be:

    - ``None``
    - integer
    - arrays

    Parameters
    ----------
    first, second : None, int, or array-like
        The two input values to compare
    rtol, atol : float, optional
        The kwargs ``rtol`` and ``atol`` are passed through to the :py:meth:`numpy.allclose` method.
    equal_nan : bool, optional
        The kwarg ``equal_nan`` is passed through to the :py:meth:`numpy.allclose` method.
    allow_one_none : bool, optional
        If set to True, will cause the function to return True if
        only one of `first` and `second` is None.

    Returns
    -------
    bool
        Whether the arguments ``first`` and ``second`` can be considered equal or not. The rules
        for doing so are as follows:
        
        - If both ``first`` and ``second`` are ``None``, then the function returns ``True``.
        - If exactly one of ``first`` and ``second`` are ``None``, and ``allow_one_none`` is
          ``True``, then the function returns ``True``.
        - If ``first`` and ``second`` are not of the same input type, then ``False`` is
          returned.
        - Otherwise, ``first`` and ``second`` are compared using the :py:meth:`np.allclose`
          method, and the return of that function is returned.
    """
    if first is None and second is None:
        return True
    if allow_one_none and ((first is None) != (second is None)):
        return True
    if isinstance(first, (list, np.ndarray)) != isinstance(second, (list, np.ndarray)):
        # Mixed input types, so no equivalence
        return False
    # Go ahead and do the comparison
    return np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=equal_nan)


def requires_training(method):
    """
    A decorator for model methods that require training before being run.

    Parameters
    ----------
    method: str
        A method name belonging to :py:class:`CannonModel`.
    """

    @wraps(method)
    def wrapper(model, *args, **kwargs):
        if not model.is_trained:
            raise TypeError("the model requires training first")
        return method(model, *args, **kwargs)

    return wrapper


class CannonModel(object):
    """
    A model for The Cannon which includes L1 regularization and pixel censoring.

    Parameters
    ----------
    training_set_labels : 2D array-like
        A set of objects with labels known to high fidelity. This can be
        given as a numpy structured array, or an astropy table. This array should
        have dimensions ``(num_stars, num_labels)``.
    training_set_flux : 2D array-like
        An array of normalised fluxes for stars in the labelled set, given
        as shape ``(num_stars, num_pixels)``. The ``num_stars`` should match the
        number of rows in ``training_set_labels``.
    training_set_ivar : 2D array-like
        An array of inverse variances on the normalized fluxes for stars in
        the training set. The shape of the ``training_set_ivar`` array should
        match that of ``training_set_flux``.
    vectorizer : subclass of :py:class:`vectorizer.base.BaseVectorizer`
        A vectorizer to take input labels and produce a design matrix. This
        should be a sub-class of :py:class:`vectorizer.base.BaseVectorizer`.
    dispersion : None, or 1D array
        The dispersion values corresponding to the given pixels. If provided,
        this should have a size of ``num_pixels``.
    regularization : ``None``, float, or 1D array
        The strength of the L1 regularization. This should either be ``None``,
        a float-type value for single regularization strength for all pixels,
        or a float-like array of length ``num_pixels``.
    censors : None, dict, or :py:class:`censoring.Censors` object
        A :py:class:`censoring.Censors` object or dictionary, containing label names
        as keys, and boolean censoring masks as values.
    """

    _data_attributes = ("training_set_labels", "training_set_flux", "training_set_ivar")

    # Descriptive attributes are needed to train *and* test the model.
    _descriptive_attributes = (
        "vectorizer",
        "censors",
        "regularization",
        "dispersion",
        "_scales",
        "_fiducials",
    )
    _computed_descriptive_attributes = ("_scales", "_fiducials")

    # Trained attributes are set only at training time.
    _trained_attributes = ("theta", "s2")

    def __init__(
        self,
        training_set_labels,
        training_set_flux,
        training_set_ivar,
        vectorizer,
        dispersion=None,
        regularization=None,
        censors=None,
        **kwargs,
    ):
        # Save the vectorizer.
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError(
                "vectorizer must be a sub-class of vectorizer.BaseVectorizer"
            )
        self._vectorizer = vectorizer

        if training_set_labels is None:
            raise ValueError("training_set_labels must not be None")

        if training_set_flux is None and training_set_ivar is None:
            # Must be reading in a model that does not have the training set
            # spectra saved.
            self._training_set_flux = None
            self._training_set_ivar = None
        else:
            self._training_set_flux = np.atleast_2d(training_set_flux)
            self._training_set_ivar = np.atleast_2d(training_set_ivar)
            # Check that the flux and ivar are valid.
            self._verify_training_data()

        if not type(training_set_labels) == np.ndarray:
            # Don't explicitly test against various types - there are a variety of
            # table-like objects that could work here
            # Simply catch & re-raise any errors that are encountered (i.e. can't look
            # up like that, invalid key/index, etc.)
            # Need it to be *exactly* and np.ndarray, because we want np.recarray
            # sent through this code block instead
            try:
                training_set_labels = np.array(
                    [training_set_labels[ln] for ln in vectorizer.label_names]
                ).T
            except (
                IndexError,
                KeyError,
                ValueError,
            ) as e:  # Probably an array, but mismatched to the other inputs
                raise ValueError(
                    "Unable to rectify training_set_labels against "
                    "given training_set_flux and training_set_ivar"
                ) from e

        # Check that the training labels are valid
        self._verify_training_labels(training_set_labels, **kwargs)
        self._training_set_labels = training_set_labels

        # Set regularization, censoring, dispersion.
        self.regularization = regularization
        self.censors = censors
        self.dispersion = dispersion

        # Set useful private attributes.
        # to training_set_labels
        __scale_labels_function = kwargs.get(
            "__scale_labels_function",
            lambda l: np.ptp(np.percentile(l, [2.5, 97.5], axis=0), axis=0),
        )
        __fiducial_labels_function = kwargs.get(
            "__fiducial_labels_function",
            lambda l: np.percentile(l, 50, axis=0),
        )

        try:
            self._scales = __scale_labels_function(self.training_set_labels)
            assert self._scales.shape == (self.training_set_labels.shape[1],)
        except TypeError as e:
            raise ValueError("__scale_labels_function must be callable")
        except AssertionError as e:
            raise ValueError(
                f"computed _scales from __scale_labels_function has the wrong shape {self._scales.shape} - should be {(self.training_set_labels.shape[1], )}"
            ) from e

        try:
            self._fiducials = __fiducial_labels_function(self.training_set_labels)
            assert self._fiducials.shape == (self.training_set_labels.shape[1],)
        except TypeError as e:
            raise ValueError("__fiducial_labels_function must be callable")
        except AssertionError as e:
            raise ValueError(
                f"computed _fiducials from __fiducial_labels_function has the wrong shape {self._fiducials.shape} - should be {(self.training_set_labels.shape[1],)}"
            ) from e

        self._design_matrix = vectorizer(
            (self.training_set_labels - self._fiducials) / self._scales
        ).T

        self.reset()

        return None

    # Representations.

    def __str__(self):
        return (
            "<{module}.{name} of {K} labels {trained} with a training set "
            "of {N} stars each with {M} pixels>".format(
                module=self.__module__,
                name=type(self).__name__,
                trained="trained" if self.is_trained else "",
                K=self.training_set_labels.shape[1],
                N=self.training_set_labels.shape[0],
                M=(
                    self.training_set_flux.shape[1]
                    if self.training_set_flux is not None
                    else "no"
                ),
            )
        )

    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(
            self.__module__, type(self).__name__, hex(id(self))
        )

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        if not (
            _compare_none_or_arrays(
                self.training_set_flux, other.training_set_flux, allow_one_none=True
            )
        ):
            return False
        if not (
            _compare_none_or_arrays(
                self.training_set_ivar, other.training_set_ivar, allow_one_none=True
            )
        ):
            return False
        if not (
            _compare_none_or_arrays(self.training_set_labels, other.training_set_labels)
        ):
            return False
        if self.vectorizer != other.vectorizer:
            return False
        if not (_compare_none_or_arrays(self.regularization, other.regularization)):
            return False
        if self.censors != other.censors:
            return False
        if not (_compare_none_or_arrays(self.dispersion, other.dispersion)):
            return False
        if not (_compare_none_or_arrays(self._scales, other._scales)):
            return False
        if not (_compare_none_or_arrays(self._fiducials, other._fiducials)):
            return False
        if not (_compare_none_or_arrays(self.theta, other.theta)):
            return False
        if not (_compare_none_or_arrays(self.s2, other.s2)):
            return False
        # Training status should be caught by the two checks above

        return True

    # Model attributes that cannot (well, should not) be changed.

    @property
    def training_set_labels(self):
        """Return the labels in the training set."""
        return self._training_set_labels100

    @property
    def training_set_flux(self):
        """Return the training set fluxes."""
        return self._training_set_flux

    @property
    def training_set_ivar(self):
        """Return the inverse variances of the training set fluxes."""
        return self._training_set_ivar

    @property
    def vectorizer(self):
        """Return the vectorizer for this model."""
        return self._vectorizer

    @property
    def design_matrix(self):
        """Return the design matrix for this model."""
        return self._design_matrix

    def _censored_design_matrix(self, pixel_index, fill_value=np.nan):
        """
        Return a censored design matrix for the given pixel index, and a mask of
        which theta values to ignore when fitting.

        Parameters
        ----------
        pixel_index : int
            The zero-indexed pixel number.

        Returns
        -------
        (censored_design_mask, excluded_values_mask)
            A two-length tuple containing the censored design mask for this
            pixel, and a boolean mask of values to exclude when fitting for
            the spectral derivatives.
        """

        if (
            not self.censors
            or self.censors is None
            or len(set(self.censors).intersection(self.vectorizer.label_names)) == 0
        ):
            return self.design_matrix

        data = (self.training_set_labels.copy() - self._fiducials) / self._scales
        for i, label_name in enumerate(self.vectorizer.label_names):
            try:
                use = self.censors[label_name][pixel_index]

            except KeyError:
                continue

            if not use:
                data[:, i] = fill_value

        return self.vectorizer(data).T

    @property
    def theta(self):
        """Return the theta coefficients (spectral model derivatives)."""
        return self._theta

    @property
    def s2(self):
        """Return the intrinsic variance (:math:`s^2`) for all pixels."""
        return self._s2

    # Model attributes that can be changed after initiation.

    @property
    def censors(self):
        """Return the wavelength censor masks for the labels."""
        return self._censors

    @censors.setter
    def censors(self, censors):
        """
        Set label censoring masks for each pixel.

        Parameters
        ----------
        censors : :py:class:`censoring.Censors` instance
            A dictionary-like object with label names as keys, and boolean arrays
            as values.
        """

        censors = {} if censors is None else censors
        if isinstance(censors, censoring.Censors):
            # Could be a censoring dictionary from a different model,
            # with different label names and pixels.
            try:
                assert (
                    censors.label_names == self.vectorizer.label_names
                ), "Censor label names != vectorizer label names"
                if self.training_set_flux is not None:
                    assert (
                        self.training_set_flux.shape[1] == censors.num_pixels
                    ), "Censor num_pixels != training_set_flux 2nd axis size"
            except AssertionError as e:
                raise ValueError(f"Bad input censor - {str(e)}") from e

            # But more likely: we are loading a model from disk.
            self._censors = censors

        elif isinstance(censors, dict):
            try:
                self._censors = censoring.Censors(
                    self.vectorizer.label_names,
                    self.training_set_flux.shape[1],
                    censors,  # Censors init should catch bad input
                )
            except AttributeError:  # training_set_flux is None:
                self._censors = None

        else:
            raise TypeError(
                "censors must be a dictionary or a censoring.Censors object"
            )

    @property
    def dispersion(self):
        """Return the dispersion points for all pixels."""
        return self._dispersion

    @dispersion.setter
    def dispersion(self, dispersion):
        """
        Set the dispersion values for all the pixels.

        Parameters:
        dispersion : 1D array
            An array of the dispersion values.
        """
        if dispersion is None:
            self._dispersion = None
            return None

        dispersion = np.array(dispersion).flatten()
        if (
            self.training_set_flux is not None
            and dispersion.size != self.training_set_flux.shape[1]
        ):
            raise ValueError(
                "dispersion provided does not match the number "
                "of pixels per star ({0} != {1})".format(
                    dispersion.size, self.training_set_flux.shape[1]
                )
            )

        if dispersion.dtype.kind not in "iuf":
            raise ValueError("dispersion values are not float-like")

        if not np.all(np.isfinite(dispersion)):
            raise ValueError("dispersion values must be finite")

        self._dispersion = dispersion
        return None

    @property
    def regularization(self):
        """Return the strength of the L1 regularization for this model."""
        return self._regularization

    @regularization.setter
    def regularization(self, regularization):
        """
        Specify the strength of the regularization for the model, either as a
        single value for all pixels, or a different strength for each pixel.

        Parameters
        ----------
        regularization : float, or 1D array
            The L1-regularization strength for the model.
        """

        if regularization is None:
            self._regularization = None
            return

        regularization = np.array(regularization).flatten()
        if regularization.size == 1:
            regularization = regularization[0]
        elif (
            self.training_set_flux is not None
            and regularization.size != self.training_set_flux.shape[1]
        ):
            raise ValueError("regularization array must be of size `num_pixels`")

        try:
            if np.any(0 > regularization) or not np.all(np.isfinite(regularization)):
                raise ValueError("regularization must be positive and finite")
        except (
            ValueError,
            TypeError,
        ) as e:  # Typically a non-numeric input has been found
            raise ValueError("regularization must be positive and finite (and numeric)")

        self._regularization = regularization
        return

    # Convenient functions and properties.

    @property
    def is_trained(self):
        """Return :py:obj:`True` or :py:obj:`False` for whether the model is trained."""
        return all(
            getattr(self, attr, None) is not None for attr in self._trained_attributes
        )

    def reset(self):
        """Clear any Model attributes that have been trained."""
        for attribute in self._trained_attributes:
            setattr(self, "_{}".format(attribute), None)
        return None

    @classmethod
    def _pixel_access(cls, array, index, default=None):
        """
        Safely access a (potentially per-pixel) attribute of the model.

        Parameters
        ----------
        array : ``None`` or float or array
            Either ``None``, a float value, or an array.
        index : int
            The zero-indexed pixel to attempt to access.
        default : optional
            The default value to return if ``array`` is None, or if an out-of-bounds index
            is requested from the array.
        """

        if array is None:
            return default
        try:
            return array[index]
        except TypeError:
            return array
        except IndexError:
            return default

    def _verify_training_data(self):
        """
        Verify the training data (``flux`` and ``ivar``) for the appropriate shape and content.
        """

        if (self.training_set_flux is None != self.training_set_ivar is None) or (
            (self.training_set_flux[0][0] is None)
            != (self.training_set_ivar[0][0] is None)
        ):
            raise ValueError(
                "training set flux and inverse variance arrays must both exist, or both be None"
            )

        if self.training_set_flux.shape != self.training_set_ivar.shape:
            print(self.training_set_flux)
            print(self.training_set_ivar)
            raise ValueError(
                "the training set flux and inverse variance arrays"
                " for the labelled set must have the same shape"
            )

        if not np.all(np.isfinite(self.training_set_flux)):
            raise ValueError("training set fluxes are not all finite")

        if not np.all(self.training_set_ivar >= 0) or not np.all(
            np.isfinite(self.training_set_ivar)
        ):
            raise ValueError("training set ivars are not all positive finite")

        return None

    def _verify_training_labels(self, training_set_labels, rho_warning=0.90, **kwargs):
        """
        Verify the training labels for the appropriate shape and context.

        Parameters
        ----------
        training_set_labels : 2D array
            The training label values array to check.
        rho_warning : :py:obj:`float`, optional
            Maximum correlation value between labels before a warning is given.
        """

        if (
            self.training_set_flux is not None
            and len(training_set_labels) != self.training_set_flux.shape[0]
        ):
            raise ValueError(
                "the first axes of the training set labels array should "
                "have the same shape as the number of rows in the labelled training set"
                "(N_stars, N_pixels)"
            )

        if training_set_labels.shape[1] != len(self.vectorizer.label_names):
            raise ValueError(
                "The second axis of the training set labels array should have the same "
                "size as the number of label names in the vectorizer"
            )

        if not np.all(np.isfinite(training_set_labels)):
            raise ValueError("training set labels are not all finite")

        # Look for very high correlation coefficients between labels, which
        # could make the training time very difficult.
        rho = np.corrcoef(training_set_labels.T)

        # Set the diagonal indices to zero.
        if len(rho.shape) > 0:
            K = rho.shape[0]
            rho[np.diag_indices(K)] = 0.0
            indices = np.argsort(rho.flatten())[::-1]

            # FIXME use array logic here for speed-up
            for index in indices:
                x, y = (index % K, int(index / K))
                rho_xy = abs(rho[x, y])
                if rho_xy >= rho_warning:
                    if x > y:  # One warning per correlated label pair.
                        logger.warning(
                            "Labels '{X}' and '{Y}' are highly correlated ("
                            "rho = {rho_xy:.2}). This may cause very slow training "
                            "times. Are both labels needed?".format(
                                X=self.vectorizer.label_names[x],
                                Y=self.vectorizer.label_names[y],
                                rho_xy=rho_xy,
                            )
                        )

        return None

    def in_convex_hull(self, labels):
        """
        Return whether the provided labels are inside a complex hull constructed
        from the labelled set.

        Parameters
        ----------
        labels : 2D array
            A `NxK` array of `N` sets of `K` labels, where `K` is the number of
            labels that make up the vectorizer.

        Returns
        -------
        2D array
            A boolean array as to whether the points are in the complex hull of
            the labelled set.
        """

        if self.training_set_labels.shape[1] == 1:
            raise RuntimeError("Cannot run in_convex_hull with a single label")

        labels = np.atleast_2d(labels)
        if labels.shape[1] != self.training_set_labels.shape[1]:
            raise ValueError(
                "expected {} labels; got {}".format(
                    self.training_set_labels.shape[1], labels.shape[1]
                )
            )

        hull = Delaunay(self.training_set_labels)
        return hull.find_simplex(labels) >= 0

    def write(
        self, path, include_training_set_spectra=False, overwrite=False, protocol=-1
    ):
        """
        Serialise the trained model and save it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        Parameters
        ----------
        path : str or :py:obj:`pathlib.Path`
            The path to save the model to.
        include_training_set_spectra : bool, optional
            Save the labelled set, normalised flux and inverse variance used to
            train the model.
        overwrite : bool, optional
            Overwrite the existing file path, if it already exists.
        protocol : int, optional
            The Python pickling protocol to employ. Use 2 for compatibility with
            previous Python releases, -1 for performance.
        """

        if os.path.exists(path) and not overwrite:
            raise IOError("path already exists: {0}".format(path))

        attributes = (
            list(self._descriptive_attributes)
            + list(self._trained_attributes)
            + list(self._data_attributes)
        )

        if "metadata" in attributes:
            logger.warning("'metadata' is a protected attribute. Ignoring.")
            attributes.remote("metadata")

        # import pdb; pdb.set_trace()

        # Store up all the trained attributes and a hash of the training set.
        state = {}
        for attribute in attributes:
            value = getattr(self, attribute)

            try:
                # If it's a vectorizer or censoring dict, etc, get the state.
                value = value.__getstate__()
            except:
                None

            state[attribute] = value

        # Create a metadata dictionary.
        state["metadata"] = dict(
            version=__version__,
            model_class=type(self).__name__,
            modified=str(datetime.now()),
            data_attributes=self._data_attributes,
            descriptive_attributes=self._descriptive_attributes,
            computed_descriptive_attributes=self._computed_descriptive_attributes,
            trained_attributes=self._trained_attributes,
            training_set_hash=utils.short_hash(
                getattr(self, attr) for attr in self._data_attributes
            ),
        )

        if not include_training_set_spectra:
            state.pop("training_set_flux")
            state.pop("training_set_ivar")

        elif not self.is_trained:
            logger.warning(
                "The training set spectra won't be saved, and this model"
                "is not already trained. The saved model will not be "
                "able to be trained when loaded!"
            )

        with open(path, "wb") as fp:
            pickle.dump(state, fp, protocol)
        return None

    @classmethod
    def read(cls, path, **kwargs):
        """
        Read a saved model from disk.

        Parameters
        ----------
        path : :py:obj:`str`, or :py:class:`pathlib.Path` object
            The path where to load the model from.
        """

        encodings = ("utf-8", "latin-1")
        for encoding in encodings:
            kwds = {"encoding": encoding} if version_info[0] >= 3 else {}
            try:
                with open(path, "rb") as fp:
                    state = pickle.load(fp, **kwds)

            except UnicodeDecodeError:
                if encoding == encodings:
                    raise

        # import pdb; pdb.set_trace()

        # Parse the state.
        metadata = state.get("metadata", {})
        version_saved = metadata.get("version", "0.1.0")
        if version_saved >= "0.2.0":  # Refactor'd.
            init_attributes = list(metadata["data_attributes"]) + list(
                metadata["descriptive_attributes"]
            )

            kwds = dict([(a, state.get(a, None)) for a in init_attributes])

            # Initiate the vectorizer.
            vectorizer_class, vectorizer_kwds = kwds["vectorizer"]
            klass = getattr(vectorizer_module, vectorizer_class)
            kwds["vectorizer"] = klass(**vectorizer_kwds)

            # Initiate the censors.
            kwds["censors"] = censoring.Censors(**kwds["censors"])

            model = cls(**kwds)
            # Computed descriptive attributes need to be set directly, as the functions
            # used to compute them in __init__ are not recorded
            for attr in metadata.get(
                "computed_descriptive_attributes", []
            ):  # Protect against old saves not defining this
                setattr(
                    model, attr, state.get(attr, getattr(model, attr))
                )  # No-op if not defined

            # Set training attributes.
            for attr in metadata["trained_attributes"]:
                setattr(model, "_{}".format(attr), state.get(attr, None))

            return model

        else:
            raise NotImplementedError(
                "Cannot auto-convert old model files yet; "
                "contact Andy Casey <andrew.casey@monash.edu> if you need this"
            )

    def train(
        self, threads=None, op_method=None, op_strict=True, op_kwds=None, **kwargs
    ):
        """
        Train the model.

        Parameters
        ----------
        threads : int, optional
            The number of parallel threads to use.
        p_method : str, optional
            The optimization algorithm to use: ``"l_bfgs_b"`` (default) and 
            ``"powell"`` are available.
        op_strict : bool, optional
            Default to Powell's optimization method if BFGS fails.
        op_kwds : bool, optional
            Keyword arguments to provide directly to the optimization function.

        Returns
        -------
        (theta, s2, metadata)
            A three-length tuple containing the spectral coefficients ``theta``,
            the squared scatter term at each pixel ``s2``, and metadata related to
            the training of each pixel.
        """

        kwds = dict(op_method=op_method, op_strict=op_strict, op_kwds=op_kwds)
        kwds.update(kwargs)

        if self.training_set_flux is None or self.training_set_ivar is None:
            raise TypeError(
                "cannot train: training set spectra not saved with the model"
            )

        S, P = self.training_set_flux.shape
        T = self.design_matrix.shape[1]

        logger.info(
            "Training {0}-label {1} with {2} stars and {3} pixels/star".format(
                len(self.vectorizer.label_names), type(self).__name__, S, P
            )
        )

        # Parallelise out.
        # TODO check current standard for parallelization
        if threads in (1, None):
            mapper, pool = (map, None)

        else:
            pool = mp.Pool(threads)
            mapper = pool.map

        func = utils.wrapper(fitting.fit_pixel_fixed_scatter, None, kwds, P)

        meta = []
        theta = np.nan * np.ones((P, T))
        s2 = np.nan * np.ones(P)

        for pixel, (flux, ivar) in enumerate(
            zip(self.training_set_flux.T, self.training_set_ivar.T)
        ):
            args = (
                flux,
                ivar,
                self._initial_theta(pixel),
                self._censored_design_matrix(pixel),
                self._pixel_access(self.regularization, pixel, 0.0),
                None,
            )
            ((pixel_theta, pixel_s2, pixel_meta),) = mapper(func, [args])

            meta.append(pixel_meta)
            theta[pixel], s2[pixel] = (pixel_theta, pixel_s2)

        self._theta, self._s2 = (theta, s2)

        if pool is not None:
            pool.close()
            pool.join()

        return (theta, s2, meta)

    @requires_training
    def __call__(self, labels):
        """
        Return spectral fluxes, given the labels.

        Parameters
        ----------
        labels : 2D array
            An array of stellar labels.
        """

        # Scale and offset the labels.
        scaled_labels = (np.atleast_2d(labels) - self._fiducials) / self._scales
        flux = np.dot(self.theta, self.vectorizer(scaled_labels)).T
        return flux[0] if flux.shape[0] == 1 else flux

    @requires_training
    def test(
        self,
        flux,
        ivar,
        initial_labels=None,
        threads=None,
        use_derivatives=True,
        op_kwds=None,
    ):
        """
        Run the test step on spectra.

        Parameters
        ----------
        flux : 2D array
            The (pseudo-continuum-normalized) spectral flux.
        ivar : 2D array
            The inverse variance values for the spectral fluxes.
        initial_labels : 2D array, optional
            The initial labels to try for each spectrum. This can be a single
            set of initial values, or one set of initial values for each star.
        threads : int, optional
            The number of parallel threads to use.
        use_derivatives : bool or func, optional
            ``True`` indicates to use analytic derivatives provided by
            the vectorizer, ``None`` to calculate on the fly, or a callable
            function to calculate your own derivatives.
        op_kwds :  dict, optional
            Optimization keywords that get passed to :py:meth:`scipy.optimize.leastsq`.

        Returns
        -------
        (array, array, dict)
            The fitted labels, the covariance array, and the metadata dictionary.
        """

        if flux is None or ivar is None:
            raise ValueError("flux and ivar must not be None")

        if op_kwds is None:
            op_kwds = dict()

        if threads in (1, None):
            mapper, pool = (map, None)

        else:
            pool = mp.Pool(threads)
            mapper = pool.map

        flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
        S, P = flux.shape

        if ivar.shape != flux.shape:
            raise ValueError("flux and ivar arrays must be the same shape")

        if initial_labels is None:
            initial_labels = self._fiducials

        initial_labels = np.atleast_2d(initial_labels)
        if initial_labels.shape[0] != S and len(initial_labels.shape) == 2:
            initial_labels = np.tile(initial_labels.flatten(), S).reshape(
                S, -1, len(self._fiducials)
            )

        args = (self.vectorizer, self.theta, self.s2, self._fiducials, self._scales)
        kwargs = dict(use_derivatives=use_derivatives, op_kwds=op_kwds)

        func = utils.wrapper(
            fitting.fit_spectrum,
            args,
            kwargs,
            S,
            message="Running test step on {} spectra".format(S),
        )

        labels, cov, meta = zip(*mapper(func, zip(*(flux, ivar, initial_labels))))

        if pool is not None:
            pool.close()
            pool.join()

        return (np.array(labels), np.array(cov), meta)

    def _initial_theta(self, pixel_index, **kwargs):
        """
        Return a list of guesses of the spectral coefficients for the given
        pixel index.

        Initial values are sourced in the following preference
        order:

            (1) a previously trained ``theta`` value for this pixel,
            (2) an estimate of ``theta`` using linear algebra,
            (3) a neighbouring pixel's ``theta`` value,
            (4) the fiducial value of ``[1, 0, ..., 0]``.

        pixel_index: int
            The zero-indexed integer of the pixel.

        Returns
        -------
        list of 2-tuples
            A list of initial theta guesses, and the source of each guess.
        """

        guesses = []

        if self.theta is not None:
            # Previously trained theta value.
            if np.all(np.isfinite(self.theta[pixel_index])):
                guesses.append((self.theta[pixel_index], "previously_trained"))

        # Estimate from linear algebra.
        theta, cov = fitting.fit_theta_by_linalg(
            self.training_set_flux[:, pixel_index],
            self.training_set_ivar[:, pixel_index],
            s2=kwargs.get("s2", 0.0),
            design_matrix=self.design_matrix,
        )
        if np.all(np.isfinite(theta)):
            guesses.append((theta, "linear_algebra"))

        if self.theta is not None:
            # Neighbouring pixels value.
            for neighbour_pixel_index in set(
                np.clip(
                    [pixel_index - 1, pixel_index + 1],
                    0,
                    self.training_set_flux.shape[1] - 1,
                )
            ):
                if np.all(np.isfinite(self.theta[neighbour_pixel_index])):
                    guesses.append(
                        (self.theta[neighbour_pixel_index], "neighbour_pixel")
                    )

        # Fiducial value.
        fiducial = np.hstack([1.0, np.zeros(len(self.vectorizer.terms))])
        guesses.append((fiducial, "fiducial"))

        return guesses
