#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities to deal with wavelength censoring.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from copy import deepcopy

__all__ = [
    "Censors",
    "create_mask",
    # "design_matrix_mask"
]

import numpy as np

from .vectorizer.base import BaseVectorizer


class Censors(dict):
    """
    A dictionary sub-class that allows for label censoring masks to be
    applied on a per-pixel basis to CannonModel objects.

    Parameters
    ----------
    label_names : iterable of str
        A list containing the label names that form the model vectorizer.
    num_pixels : int
        The number of pixels per star.
    items : dict, optional
        A dictionary containing label names as keys and masks as values.
    """

    def __init__(self, label_names, num_pixels, items=None, **kwargs):
        super(Censors, self).__init__(**kwargs)
        self._label_names = list(label_names)
        self._num_pixels = int(num_pixels)
        self.update(items or {})
        return None

    def __setitem__(self, label_name, mask):
        """
        Update an entry in the pixel censoring dictionary.

        Parameters
        ----------
        label_name: str
            The name of the label to apply the censoring to.
        mask: array-like of bool
            A boolean mask with a size that equals the number of pixels per star.
            Note that a mask value of `True` indicates the label is censored at
            the given pixel, and therefore that label will not contribute to
            the spectral flux at that pixel.
        """

        if label_name not in self.label_names:
            raise ValueError(
                "unrecognized label name '{}' for censoring".format(label_name)
            )

        mask = np.array(mask).flatten().astype(bool)
        if mask.size != self.num_pixels:
            raise ValueError(
                "'{}' censoring mask has wrong size ({} != {})".format(
                    label_name, mask.size, self.num_pixels
                )
            )

        dict.__setitem__(self, label_name, mask)
        return None

    def __eq__(self, other):
        if self.num_pixels != other.num_pixels:
            return False
        if self.label_names != other.label_names:
            return False
        # Need to do this manually, otherwise get ambiguous truth value in array error
        if (
            self.keys() != other.keys()
        ):  # Probably redundant, but safest to include as backup
            return False
        for k in self.keys():
            if np.any(self[k] != other[k]):
                return False
        return True

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.update(self)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        result.update(self)
        return result

    def update(self, *args, **kwargs):
        """Update with a second :py:obj:`Censors` dict."""
        if args:
            if len(args) > 1:
                raise TypeError(
                    "update expected at most 1 arguments, got {}".format(len(args))
                )
            other = dict(args[0])
            for key in other:
                self[key] = other[key]

        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        """Set default value"""
        if key not in self:
            self[key] = value
        return self[key]

    def __getstate__(self):
        """Return the state of the censoring mask in a serializable form."""
        return dict(
            label_names=self.label_names,
            num_pixels=self.num_pixels,
            items=dict(self.items()),
        )

    @property
    def label_names(self):
        """The label names for this Censors."""
        return self._label_names

    @property
    def num_pixels(self):
        """The num_pixels for this Censors."""
        return self._num_pixels


def create_mask(dispersion, censored_regions):
    """
    Return a boolean censoring mask based on a structured list of (start, end)
    regions.

    Parameters
    ----------
    dispersion : array-like of floats
        An array of dispersion values.
    censored_regions : list of two-tuples
        A list of two-length tuples containing the `(start, end)` points of a
        censored region.

    Returns
    -------
    array-like of bool
        A boolean mask indicating whether the pixels in the `dispersion` array
        are masked.
    """

    mask = np.zeros(dispersion.size, dtype=bool)

    if len(censored_regions) == 0:  # No-op
        return mask

    if isinstance(censored_regions[0], (int, float)):
        censored_regions = [censored_regions]

    for start, end in censored_regions:
        start, end = (start or -np.inf, end or +np.inf)

        try:
            censored = (end >= dispersion) * (dispersion >= start)
        except (np.core._exceptions._UFuncNoLoopError, TypeError) as e:
            raise ValueError(
                "Encountered error in computing censored array, " \
                "likely bad censored_region value type"
            )
        mask[censored] = True

    return mask
