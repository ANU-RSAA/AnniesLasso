#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A polynomial vectorizer for The Cannon.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

__all__ = [
    "PolynomialVectorizer",
    "parse_label_vector_description",
    "human_readable_label_term",
    "human_readable_label_vector",
    "terminator",
    "get_label_names",
]

import numpy as np
from collections import Counter, OrderedDict
from itertools import combinations_with_replacement

from .base import BaseVectorizer


class PolynomialVectorizer(BaseVectorizer):
    """
    A vectorizer that models spectral fluxes as combination of polynomial terms.

    Note that either ``label_names`` *and* ``order`` must be provided, or the
    ``terms`` keyword argument needs to be explicitly specified.

    Parameters
    ----------
    label_names : iterable of str, optional
        A list of label names that are terms in the label vector.

    order : int, optional
        The maximal order for the vectorizer.

    terms : list of list of 2-tuples, optional
        A structured list of tuples that defines the full extent of the label
        vector. Note that ``terms`` *must* be ``None`` if ``label_names``
        and ``order`` are provided.

        The terms list should be structured as for
        :py:class:`thecannon.vectorizer.base.BaseVectorizer`.
    """

    def __init__(self, label_names=None, order=None, terms=None, **kwargs):
        # Check to see if we have a terms/(label_names and order) dichotamy/
        if (terms is None and None in (label_names, order)) or (
            terms is not None and order is not None
        ):
            raise ValueError(
                "order must be None if terms are provided, "
                "and terms must be None if label_names and order are provided"
            )

        if terms is None:
            # Parse human-readable terms.
            terms = terminator(label_names, order, **kwargs)
        elif label_names is None:
            # Parse label names from the terms.
            label_names = get_label_names(parse_label_vector_description(terms))

        # Convert terms from a string to standard structure
        if not isinstance(terms, list):
            terms = parse_label_vector_description(terms, label_names=label_names)

        # FIXME work out if, by this stage, we should have settled on labels or
        # ints in the terms vector
        # Currently seem to be on label strs at this point
        super(PolynomialVectorizer, self).__init__(
            label_names=label_names, terms=terms, **kwargs
        )
        self.index_labels()
        return None

    def index_labels(self):
        """Re-compile `self.terms` to use indices.

        This function can be run regardless of the current state of `self.terms`
        (i.e., whether or not `self.terms` currently uses indices).

        Raises
        ------
        ValueError
            If `self.terms` and/or `self.label_names` are not set.
        """

        if self.terms is None or self.label_names is None:
            raise ValueError("terms and/or label_names haven't been set!")
        terms = parse_label_vector_description(
            human_readable_label_vector(
                self.terms, const=False, label_names=self.label_names
            ),
            label_names=self.label_names,
        )
        self.update_labels_terms(self.label_names, terms)

    def get_label_vector(self, labels):
        """
        Return the values of the label vector, given the scaled labels.

        This function computes term-by-term values of the model terms, for the input
        vector of individual label values/input array of label values for a number of
        test cases.

        Parameters
        ----------
        labels : 1D or 2D array
            The scaled and offset labels to use to calculate the label vector(s).
            This can be a ond-dimensional vector of ``K`` labels, or a
            two-dimensional array of ``N`` by ``K`` labels.

            Note that ``K`` must *exactly* match the number of labels in the vectorizer.
            This is because the labels are mapped using their position (index) in the
            ``labels`` input. Therefore, having ``K != len(self.label_names)`` means there
            is ambiguity in the mapping of label names to input label values. (Note that
            this also means that ``labels`` must be ordered in the same way as, e.g.,
            ``self.label_names``, but there is no way to test/enforce this.)

        Returns
        -------
        2D array
            The returning array will be of shape ``(N, K+1)``,
            where ``K`` is the number of terms in the label vector description. The extra term
            (which is always a representation of "+1") stores the model mean.
        """

        labels = np.atleast_2d(labels)
        if labels.ndim > 2:
            raise ValueError("labels must be a 1-d or 2-d array")

        if self.terms is None:
            raise RuntimeError(
                "Vectorizer terms need to be set before calling get_label_vector"
            )

        # Sanity check the inputs
        try:
            assert labels.shape[1] == len(self.label_names)
        except AssertionError as e:
            raise ValueError(
                f"Your array of `N` ({labels.shape[0]}) training values must each have {len(self.label_names)} labels assigned"
            ) from e

        columns = [np.ones(labels.shape[0], dtype=float)]
        for term in self.terms:
            column = 1.0  # This works; don't use np.multiply/np.product.
            for index, order in term:
                column *= labels[:, index] ** order
            columns.append(column)
        return np.vstack(columns)

    def get_label_vector_derivative(self, labels):
        """
        Return the derivatives of the label vector with respect to fluxes.

        Parameters
        ----------
        labels : 1D array
            The scaled labels to calculate the label vector derivatives. This should
            be a one-dimensional vector of ``K`` labels (using the same order and
            length provided by ``self.label_names``).

        Returns
        -------
        2D array
            The returning array will be of shape ``(N, D)``,
            where ``D`` is the number of terms in the label vector description.
        """

        N, T = (len(labels), len(self.terms))
        if np.asarray(labels).ndim != 1:
            raise ValueError(
                "get_label_vector_derivative only accepts 1D arrays/lists for labels"
            )
        if N != len(self.label_names):
            raise ValueError(
                f"Number of input label values {N} does not match number of vectorizer labels {T}"
            )

        slicer = np.arange(N)
        indices_used = np.zeros(N, dtype=bool)

        columns = np.ones((T + 1, N), dtype=float)
        columns[0] = 0.0  # First theta derivative always zero.

        for t, term in enumerate(self.terms, start=1):
            indices_used[:] = False

            for index, order in term:
                dy = order * (labels[index] ** (order - 1))
                y = labels[index] ** order

                # If it's the index w.r.t. it, take derivative.
                columns[t, index] *= dy

                # Otherwise, calculate as normal.
                columns[t, slicer != index] *= y
                indices_used[index] = True

            columns[t, ~indices_used] = 0

        return columns

    def get_human_readable_label_vector(self, mul="*", pow="^", bracket=False):
        """
        Return a human-readable form of the label vector.

        Parameters
        ----------
        mul : str, optional
            String to use to represent a multiplication operator. For example,
            if giving LaTeX label definitions one may want to use ``"\\cdot"`` for
            the ``mul`` term.

        pow : str, optional
            String to use to represent a power operator.

        bracket : bool, optional
            Show brackets around each term.

        Returns
        -------
        str
            A human-readable string representing the label vector.
        """
        return human_readable_label_vector(
            self.terms, self.label_names, mul=mul, pow=pow, bracket=bracket
        )

    @property
    def human_readable_label_vector(self):
        """
        Return a human-readable form of the label vector.
        """
        return self.get_human_readable_label_vector()

    def get_human_readable_label_term(self, term_index, label_names=None, **kwargs):
        """
        Return a human-readable form of a single term in the label vector.

        Parameters
        ----------
        term_index : int
            The term in the label vector to return.

        label_names : list of str, optional
            The label names to use. For example, these could be LaTeX
            representations of the label names.

        Returns
        -------
        str
            A human-readable string representing a single term in the label vector.
        """

        if term_index == 0:
            return "1"
        else:
            return human_readable_label_term(
                self.terms[term_index - 1],
                label_names=label_names or self.label_names,
                **kwargs,
            )


def _is_structured_label_vector(label_vector):
    """
    Return whether the provided label vector is structured as a polynomial
    vector description appropriately or not.

    Parameters
    ----------
    label_vector: str
        A structured or unstructured description of a polynomial label vector.
    """
    if not isinstance(label_vector, (list, tuple)):
        return False

    for descriptor in label_vector:
        if not isinstance(descriptor, (list, tuple)):
            return False

        for term in descriptor:
            if (
                not isinstance(term, (list, tuple))
                or len(term) != 2
                or not isinstance(term[-1], (int, float))
            ):
                return False

    if len(label_vector) == 0 or sum(map(len, label_vector)) == 0:
        return False
    return True


def parse_label_vector_description(description, label_names=None, **kwargs):
    """
    Return a structured form of a label vector from unstructured,
    human-readable input.

    Parameters
    ----------
    description : str or list
        A human-readable or structured form of a label vector.
    label_names : list of str, optional
        If ``label_names`` are provided, instead of label names being provided as
        the output parameter, the corresponding index location will be given.

    Returns
    -------
    list
        A structured form of the label vector as a multi-level list.


    Examples
    --------
    For example, to get a representation of :math:`Teff^4 + logg * Teff^3 + feh + feh^0 * Teff`::

        >>> parse_label_vector_description("Teff^4 + logg*Teff^3 + feh + feh^0*Teff")
        >>> [
                [
                    ("Teff", 4),
                ],
                [
                    ("logg", 1),
                    ("Teff", 3)
                ],
                [
                    ("feh", 1),
                ],
                [
                    ("feh", 0),
                    ("Teff", 1)
                ]
            ]
    """

    if _is_structured_label_vector(description):
        return description

    # Allow for custom characters, but don't advertise it.
    # (Astronomers have bad enough habits already.)
    kwds = dict(zip(("sep", "mul", "pow"), "+*^"))
    kwds.update(kwargs)
    sep, mul, pow = (kwds[k] for k in ("sep", "mul", "pow"))

    if isinstance(description, (str,)):
        description = description.split(sep)
    description = [_.strip() for _ in description]

    # Functions to parse the parameter (or index) and order for each term.
    get_power = lambda t: int(t.split(pow)[1].strip()) if pow in t else 1
    if label_names is None:
        get_label = lambda d: d.split(pow)[0].strip()
    else:
        get_label = lambda d: list(label_names).index(d.split(pow)[0].strip())

    label_vector = []
    for descriptor in (item.split(mul) for item in description):
        labels = map(get_label, descriptor)
        orders = map(get_power, descriptor)

        term = OrderedDict()
        for label, order in zip(labels, orders):
            term[label] = term.get(label, 0) + order  # Sum repeat term powers.

        # Prevent uses of x^0 etc clogging up the label vector.
        # MCW - also prevent the empty string getting through as a label name
        valid_terms = [(l, o) for l, o in term.items() if o != 0 and l != ""]
        if not np.all(np.isfinite([o for l, o in valid_terms])):
            raise ValueError("non-finite power provided")

        if len(valid_terms) > 0:
            label_vector.append(valid_terms)

    if sum(map(len, label_vector)) == 0:
        raise ValueError("no valid terms provided")

    return label_vector


def human_readable_label_term(term, label_names=None, mul="*", pow="^", bracket=False):
    """
    Return a human-readable form of a single term in the label vector.

    Parameters
    ----------
    term : list
        A structured term.
    label_names : list of str, optional
        The names for each label in the label vector.
    mul : str, optional
        String to use to represent a multiplication operator. For example,
        if giving LaTeX label definitions one may want to use ``"\\cdot"`` for
        the ``mul`` term.
    pow : str, optional
        String to use to represent a power operator.
    bracket : bool, optional
        Show brackets around each term.

    Returns
    -------
    str
        A human-readable string representing the label vector.
    """
    # Input checking
    try:
        assert _is_structured_label_vector(
            [
                term,
            ]
        )
    except AssertionError:
        raise ValueError(f"{term} is not a valid structured term")

    ct = []
    for i, o in term:
        if isinstance(i, int) and label_names is not None:
            label_name = label_names[i]
        elif isinstance(i, int) and label_names is None:
            raise ValueError("Need label_names to convert indexed labels to strings")
        else:
            label_name = i
        if o > 1:
            d = (0, 1)[o - int(o) > 0]
            ct.append("{0}{1}{2:.{3}f}".format(label_name, pow, o, d))
        else:
            ct.append(label_name)

    if bracket and len(ct) > 1:
        return "({})".format(mul.join(ct))
    else:
        return mul.join(ct)


def human_readable_label_vector(
    terms, label_names=None, mul="*", pow="^", sep="+", bracket=False, const=True
):
    """
    Return a human-readable form of the label vector.

    Parameters
    ----------
    terms : list
        The structured terms of the label vector.
    label_names : list of str, optional
        The names for each label in the label vector.
    mul : str, optional
        String to use to represent a multiplication operator. For example,
        if giving LaTeX label definitions one may want to use ``"\\cdot"`` for
        the ``mul`` term.
    pow : str, optional
        String to use to represent a power operator.
    bracket : bool, optional
        Show brackets around each term.
    const : bool, optional
        Add a constant "1 +" to the start of the label vector.

    Returns
    -------
    str
        A human-readable string representing the label vector.
    """
    if not _is_structured_label_vector(terms):
        raise ValueError("label vector is not a structured set of terms")

    human_terms = []
    if const:
        human_terms = ["1"]
    for term in terms:
        human_terms.append(
            human_readable_label_term(
                term, label_names=label_names, mul=mul, pow=pow, bracket=bracket
            )
        )
    return f" {sep} ".join(human_terms)


def terminator(label_names, order, cross_term_order=-1, **kwargs):
    """
    Create the terms required for a label vector description based on the label
    names provided and the order given.

    Parameters
    ----------
    label_names : list of str
        The names of the labels to use in describing the label vector.
    order : int
        The maximum order of the terms (e.g., order ``3`` implies :math:`A^3` is a term).
    cross_term_order : int, optional
        The maximum order of the cross-terms (e.g., cross_term_order ``2`` implies
        :math:`A^2*B` is a term). If the provided ``cross_term_order`` value is negative,
        then ``cross_term_order = order - 1`` will be assumed.
    mul : str, optional
        The operator to use to represent multiplication in the description of
        the label vector.
    pow : str, optional
        The operator to use to represent exponents in the description of the
        label vector.

    Returns
    -------
    str
        A human-readable form of the label vector.
    """

    sep, mul, pow = (
        kwargs.pop(k, d) for (k, d) in [("sep", "+"), ("mul", "*"), ("pow", "^")]
    )

    # I make no apologies: it's fun to code like this for short complex functions
    items = []
    if 0 > cross_term_order:
        cross_term_order = order - 1

    for o in range(1, 1 + max(order, 1 + cross_term_order)):
        for t in map(Counter, combinations_with_replacement(label_names, o)):
            # Python 2 and 3 behave differently here, so generate an ordered
            # dictionary based on sorting the keys.
            t = OrderedDict([(k, t[k]) for k in sorted(t.keys())])
            if (
                len(t) == 1
                and order >= max(t.values())
                or len(t) > 1
                and cross_term_order >= max(t.values())
            ):
                c = [pow.join([[l], [l, str(p)]][p > 1]) for l, p in t.items()]
                if c:
                    items.append(mul.join(map(str, c)))
    return " {} ".format(sep).join(items)


def get_label_names(label_vector):
    """
    Return the label names that contribute to the structured label vector
    description provided.

    Parameters
    ----------
    label_vector : list
        A structured description of the label vector.

    Returns
    -------
    list of str
        A list of the label names that make up the label vector.
    """
    return list(
        OrderedDict.fromkeys(
            [label for term in label_vector for label, power in term if power != 0]
        )
    )
