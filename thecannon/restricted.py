#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A restricted Cannon model where bounds are placed on theta coefficients in order
to make the model more physically realistic and limit information propagated
through abundance correlations.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ["RestrictedCannonModel"]

import logging
from .model import CannonModel
import numpy as np

logger = logging.getLogger(__name__)


class RestrictedCannonModel(CannonModel):
    """
    A model for The Cannon which includes L1 regularization, pixel censoring,
    and is capable of placing bounds on theta coefficients in order to make the
    model more physically realistic and limit information propagated through
    abundance correlations.

    This model is a subclass of :py:class:`thecannon.model.CannonModel`.

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
    theta_bounds : dict, optional
        A dictionary containing label names as keys and two-length tuples as
        values, indicating acceptable minimum and maximum values. Specify
        `None` to indicate no limit on a boundary.  For example::

                theta_bounds={"FE_H": (None, 0), "TEFF^3": (None, None)}
    """

    # Need to expand descriptive_attributes for this model
    _descriptive_attributes = (
        "vectorizer",
        "censors",
        "regularization",
        "dispersion",
        "_scales",
        "_fiducials",
        "theta_bounds",
    )

    def __init__(
        self,
        training_set_labels,
        training_set_flux,
        training_set_ivar,
        vectorizer,
        dispersion=None,
        regularization=None,
        censors=None,
        theta_bounds=None,
        **kwargs,
    ):

        super(RestrictedCannonModel, self).__init__(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vectorizer,
            dispersion=dispersion,
            regularization=regularization,
            censors=censors,
            **kwargs,
        )

        self.theta_bounds = theta_bounds
        return None

    def __eq__(self, other):
        if not (super().__eq__(other)):
            return False

        if self.theta_bounds != other.theta_bounds:
            return False

        return True

    @property
    def theta_bounds(self):
        """Return the boundaries placed on theta coefficients."""
        return self._theta_bounds

    @theta_bounds.setter
    def theta_bounds(self, theta_bounds):
        """
        Set lower and upper boundaries on specific theta coefficients.
        """
        theta_bounds = {} if theta_bounds is None else theta_bounds
        if isinstance(theta_bounds, dict):

            label_vector = self.vectorizer.human_readable_label_vector
            terms = label_vector.split(" + ")
            checked_bounds = {}
            for term in theta_bounds.keys():
                try:
                    bounds = tuple(theta_bounds[term])
                except TypeError:
                    raise ValueError("bounds must be in tuple-like form")
                term = str(term)

                if term not in terms:
                    logging.warning(
                        "Boundary on term '{}' ignored because it is "
                        "not in the label vector: {}".format(term, label_vector)
                    )
                else:
                    if len(bounds) != 2:
                        raise ValueError("bounds must be a two-length tuple")
                    if None not in bounds and bounds[1] < bounds[0]:
                        raise ValueError("bounds must be in (min, max) order")

                    checked_bounds[term] = bounds

            self._theta_bounds = checked_bounds

        else:
            raise TypeError("theta_bounds must be a dictionary-like object")
        
    def generate_theta_bounds(self, label_bounds):
        """
        Generate `theta_bounds` from a list (or similar) of term bounds.

        Parameters
        ----------
        label_bounds : dict
            The label limits to apply, e.g.::
            
                label_bounds = {"T": (None, 1500.0), "Fe_H": (0.1, np.inf)}

            Where no lower/upper limit applies, ``None`` or ``(+/-)np.inf`` may be used.
            ``None`` is considered a safer option, as the function will safely convert that
            to ``(+/-)np.inf``, depending on which bound(s) is ``None``.

            If a label does not appear in ``label_bounds``, it will be assumed to have no 
            limits, and no limits will be applied to any term (theta) containing that label.
            Labels that do not appear in any term will be ignored.

        Raises
        ------
        ValueError
            If a bounds tuple contains a non-numeric value, or if a minimum bound is found to 
            be greater than a maximum bound.

            The latter is to catch the situation where a bound of, e.g., ``(np.inf, <some number>)`` is
            supplied, when the user really meant ``(-np.inf, <some number>)``.
        """
        # Input checking
        if not isinstance(label_bounds, dict):
            raise ValueError("label_bounds must be a dict")
        for label, bounds in label_bounds.items():
            try:
                assert bounds[0] < bounds[1]
            except AssertionError:  # Bounds reversed
                raise ValueError(f"Bounds for label '{label}' appear to be reversed")
            except (TypeError, IndexError):  # Bounds are formatted wrong
                raise ValueError(f"Bounds for label '{label}' do not appear to be a two-tuple, "
                                 f"or may contain non-numeric values")
            # Deal with any None bounds
            # (None, None) -> unbound, remove
            # (None, *) -> -np.inf
            # (*, None) -> np.inf
            if bounds == (None, None):
                del(label_bounds[label])
            elif bounds[0] is None:
                label_bounds[label] = (-np.inf, bounds[1])
            elif bounds[1] is None:
                label_bounds[label] = (bounds[0], np.inf)

            
        theta_bounds = {}

        # Now have to go through every term in the model, and compute the min and max bounds
        label_vector = self.vectorizer.human_readable_label_vector
        terms = label_vector.split(" + ")
        for term in terms:
            # Split the term into constituent parts
            components = {l: int(p) for l, p in [_split_pow(_, "^") for _ in term.split("*")]}
            # Check if all term components are bound - if not, continue
            if not all([_ in label_bounds.keys for _ in components.keys()]):
                logger.debug(f"Not all components of term {term} are bound - skipping")
                continue

            # Construct an array of test values
            values = np.ones((2, ) * len(components))
            for i, (l, p) in enumerate(components.items()):
                for j in [0, 1]:  # 0 = min, 1 = max
                    values[(slice(None, None), ) * i + (j, ) + (slice(None, None), ) * (len(components) - i - 1)] *= label_bounds[l][j] ** p

            # Doing 0 * np.inf returns a nan - for bounds purposes, these should be returned to 0
            values[values == np.nan] = 0

            # Set the term limits based off the values array
            theta_bounds[term] = (np.min(values), np.max(values))

        self.theta_bounds = theta_bounds
        return


    def train(self, threads=None, op_kwds=None):
        """
        Train the model.

        Parameters
        ----------
        threads : int, optional
            The number of parallel threads to use.
        op_kwds : bool, optional
            Keyword arguments to provide directly to the optimization function.

        Returns
        -------
        (theta, s2, metadata)
            A three-length tuple containing the spectral coefficients ``theta``,
            the squared scatter term at each pixel ``s2``, and metadata related to
            the training of each pixel.
        """

        # Generate the optimization bounds based on self.theta_bounds.
        op_bounds = [
            self.theta_bounds.get(term, (None, None))
            for term in self.vectorizer.human_readable_label_vector.split(" + ")
        ]

        kwds = dict(op_method="l_bfgs_b", op_strict=False, op_kwds=(op_kwds or {}))
        kwds["op_kwds"].update(bounds=op_bounds)

        return super(RestrictedCannonModel, self).train(threads=threads, **kwds)
    
def _split_pow(s, c, extra="1"):
        """
        Split string by character, appending an extra value to the return of the split
        if the standard result is length 1.

        Parameters
        ----------
        str : str
            String to split
        c : str
            char/str to split s by
        extra : str, optional
            String to append to the split result if it is only length 1
        """
        r = s.split(c)
        if len(r) == 1:
            r.append(extra)
        return r
