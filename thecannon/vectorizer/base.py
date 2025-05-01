#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A base vectorizer for The Cannon.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = ["BaseVectorizer"]

import numpy as np
from typing import Union
import copy


class BaseVectorizer(object):
    """
    A vectorizer that models spectral fluxes as combination of polynomial terms.
    Note that either `label_names` *and* `order` must be provided, or the `terms`
    keyword argument needs to be explicitly specified.

    Parameters
    ----------

    label_names: list of str
        A list of label names that are terms in the label vector.

    terms: list
        A structured list of lists of tuples that defines the full extent of the label
        vector.

        The list describes the terms of the label vector as follows:

        ```
        [[[(<label index>, <label power>), ...], [(<label index>, <label power>), ...)], ...], ...]
        ```

        So, for example, if `label_names=['a', 'b']`, then the following element in `terms`:
        
        ```
        [[(0, 1), (1, 1)], [(1, 2)]]
        ```

        is equivalent to `a^1 * b^1 + b^2`. The actual label names can also be used as the first
        element of each tuple.
    """

    def __init__(
        self,
        label_names: list[str],
        terms: list[list[tuple[Union[int, str], int]]],
        **kwargs,
    ):
        if label_names is None:
            label_names = []
        if terms is None:
            terms = []
        self.update_labels_terms(tuple(label_names), terms)
        self.metadata = kwargs.get("metadata", {})
        return None
    
    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        if np.all(self.label_names != other.label_names):
            return False
        if self.terms != other.terms:
            return False
        
        return True

    # These can be over-written by sub-classes, but it is useful to have some
    # basic information if the sub-classes do not overwrite it.
    def __str__(self):
        return "<{module}.{name} object consisting of {K} labels and {D} terms>".format(
            module=self.__module__,
            name=type(self).__name__,
            D=len(self.terms),
            K=len(self.label_names),
        )

    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(
            self.__module__, type(self).__name__, hex(id(self))
        )

    # I/O (Serializable) functionality.
    def __getstate__(self):
        """Return the state of the vectorizer."""
        return (
            type(self).__name__,
            dict(
                label_names=self.label_names, terms=self.terms, metadata=self.metadata
            ),
        )

    def __setstate__(self, state):
        """Set the state of the vectorizer."""
        model_name, kwds = state
        self._label_names = kwds["label_names"]
        self._terms = kwds["terms"]
        self.metadata = kwds["metadata"]

    # FIXME add co-dependent setters for label_names, terms
    def update_labels_terms(
        self, label_names: list[str], terms: list[list[tuple[Union[int, str], int]]]
    ) -> None:
        # Sanity-check the inputs

        # Ensure that all of the label references (name or index) are valid
        # FIXME surely this can be a list comprehension...
        label_refs = set()
        powers = set()
        for l in terms:
            for t in l:
                label_refs.add(t[0])
                powers.add(int(t[1]))

        if len(label_refs) > len(label_names):
            raise ValueError(
                f"Your `terms` contain more labels than are in `label_names`."
            )
        elif len(label_names) > len(label_refs):
            raise ValueError("Your `label_names` contains unused labels.")

        # We now know that the number of labels is the same between the two arguments.
        # Now to check if they are all reasonable...
        try:
            assert np.all([_ in label_names for _ in label_refs if isinstance(_, str)])
        except AssertionError:
            raise ValueError(
                f"Unknown labels {[_ for _ in label_refs if isinstance(_, str) and _ not in label_names]} in terms"
            )

        try:
            assert np.all(
                [-1 < _ < len(label_names) for _ in label_refs if isinstance(_, int)]
            )
        except AssertionError:
            raise ValueError(
                f"Index terms references must be between 0 and {len(label_names)}"
            )

        # Don't check the powers in the terms, except to make sure that there's no power
        # 0 in there - that would be meaningless
        try:
            assert ~np.any(np.isclose(list(powers), 0))
        except AssertionError:
            raise ValueError("0th-power terms are not permitted.")

        # Make the settings
        self._label_names = list(label_names)
        self._terms = copy.deepcopy(terms)

    @property
    def terms(self):
        """Return the terms provided for this vectorizer."""
        return self._terms

    @terms.setter
    def terms(self, v):
        raise RuntimeError("terms must be set using update_labels_terms")

    @property
    def label_names(self):
        """
        Return the label names that are used in this vectorizer.
        """
        return self._label_names

    @label_names.setter
    def label_names(self, v):
        raise RuntimeError("label_names must be set using update_labels_terms")

    def __call__(self, *args, **kwargs):
        """
        An alias to the get_label_vector method.
        """
        return self.get_label_vector(*args, **kwargs)

    def get_label_vector(self, labels, *args, **kwargs):
        """
        Return the label vector based on the labels provided.

        Parameters
        ----------
        labels: list
            The values of the labels. These should match the length and order of
            the `label_names` attribute.
        """
        raise NotImplementedError(
            "the get_label_vector method " "must be specified by the sub-classes"
        )

    def get_label_vector_derivative(self, labels, *args, **kwargs):
        """
        Return the derivative of the label vector with respect to the given
        label.

        Parameters
        ----------
        labels: iterable
            The values of the labels to calculate the label vector for.
        """
        raise NotImplementedError(
            "the get_label_vector_derivative method "
            "must be specified by the sub-classes"
        )
