#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from thecannon import model
from thecannon.vectorizer.base import BaseVectorizer
from thecannon.vectorizer.polynomial import PolynomialVectorizer
from thecannon.censoring import Censors
from unittest import mock

import numpy as np


@pytest.mark.parametrize("trained", [True, False])
def test_requires_training_decorator(trained):
    class TestObject(object):
        def __init__(self, *args, **kwargs):
            self.is_trained = trained

        @model.requires_training
        def test_method(self):
            return "Test method return"

    t = TestObject()

    if trained:
        assert t.test_method() == "Test method return", "Trained return incorrect"
    else:
        with pytest.raises(TypeError, match="requires training"):
            _ = t.test_method()


@pytest.mark.parametrize(
    "vec_bad",
    [
        None,
        "a",
        1,
        1.0,
        13.5,
        "BaseVectorizer",
        BaseVectorizer,  # class, not instance
    ],
)
def test_cannonmodel_vectorizer_bad(vec_bad):
    with pytest.raises(TypeError):
        _ = model.CannonModel(np.ones(10), np.ones(10), np.ones(10), vec_bad)


@pytest.mark.parametrize(
    "vectorizer",
    [
        # BaseVectorizer,
        PolynomialVectorizer,
    ],
)
@pytest.mark.parametrize(
    "label_names,terms",
    [
        (["a"], [[("a", 2)]]),
        (["a", "b"], [[("a", 2)], [("a", 1), ("b", 1)], [("b", 3)]]),
    ],
)
class TestCannonModelInit:
    @pytest.mark.parametrize(
        "training_set_flux_shape,training_set_ivar_shape,training_set_labels_shape,error_match",
        [
            ((10,), (15,), 15, "flux and inverse"),
            ((15,), (10,), 15, "flux and inverse"),
            ((15, 2), (15, 3), 15, "flux and inverse"),
            ((10,), (15,), 10, "flux and inverse"),
            ((15,), (10,), 10, "flux and inverse"),
            ((15, 2), (15, 3), 15, "flux and inverse"),
            ((10,), (10,), 15, "the first axes"),
            ((15, 3), None, 15, "both be None"),
            (None, (15, 3), 15, "both be None"),
            ((15, 3), None, 15, "both be None"),
            (None, (15, 3), 15, "both be None"),
        ],
    )
    def test_cannonmodel_init_mismatched_training_sets(
        self,
        vectorizer,
        label_names,
        terms,
        training_set_flux_shape,
        training_set_ivar_shape,
        training_set_labels_shape,
        error_match,
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = (
            np.ones(training_set_flux_shape)
            if training_set_flux_shape is not None
            else None
        )
        training_set_ivar = (
            np.ones(training_set_ivar_shape)
            if training_set_ivar_shape is not None
            else None
        )
        training_set_labels = (
            np.ones((training_set_labels_shape, len(label_names)))
            if training_set_labels_shape is not None
            else None
        )

        with pytest.raises(ValueError, match=error_match):
            _ = model.CannonModel(
                training_set_labels, training_set_flux, training_set_ivar, vec
            )

    @pytest.mark.parametrize(
        "training_set_shape", [(10,), (10, 10), (100,), (100, 100), (10, 100)]
    )
    @pytest.mark.parametrize("inf_posn", [0, 25, 50, 75, 99])
    class TestCannonModelInfValues(object):
        def test_cannonmodel_inf_flux(
            self, vectorizer, label_names, terms, training_set_shape, inf_posn
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            training_set_flux = np.ones(training_set_shape)
            training_set_flux[
                int((inf_posn / 100.0) * training_set_flux.shape[0])
            ] = np.inf
            training_set_ivar = np.ones(training_set_shape)
            training_set_labels = np.ones((training_set_shape[0], len(label_names)))

            with pytest.raises(ValueError, match="fluxes.*finite"):
                _ = model.CannonModel(
                    training_set_labels, training_set_flux, training_set_ivar, vec
                )

        def test_cannonmodel_inf_ivar(
            self, vectorizer, label_names, terms, training_set_shape, inf_posn
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            training_set_flux = np.ones(training_set_shape)
            training_set_ivar = np.ones(training_set_shape)
            training_set_ivar[
                int((inf_posn / 100.0) * training_set_ivar.shape[0])
            ] = np.inf
            training_set_labels = np.ones(training_set_shape)

            with pytest.raises(ValueError, match="ivars.*finite"):
                _ = model.CannonModel(
                    training_set_labels, training_set_flux, training_set_ivar, vec
                )

    def test_cannonmodel_no_training_set_labels(self, vectorizer, label_names, terms):
        vec = vectorizer(label_names=label_names, terms=terms)
        with pytest.raises(ValueError):
            _ = model.CannonModel(None, None, None, vec)

    @pytest.mark.parametrize("training_labels_length", [10, 100, 1000])
    def test_cannonmodel_blank_flux_and_ivar(
        self, vectorizer, label_names, terms, training_labels_length
    ):
        training_labels = np.ones((training_labels_length, len(label_names)))
        vec = vectorizer(label_names=label_names, terms=terms)
        m = model.CannonModel(training_labels, None, None, vec)
        assert (
            m.training_set_flux is None
        ), "training set flux was set without being given"
        assert (
            m.training_set_ivar is None
        ), "training set ivar was set without being given"
        assert np.all(
            m.training_set_labels == training_labels
        ), "training set labels were incorrectly modified"

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_censoring_dict_input(
        self, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        censors = {l: np.zeros((training_shape,), dtype=bool) for l in label_names}

        m = model.CannonModel(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            censors=censors,
        )
        assert (
            set(m.censors.keys())
            == set(Censors(label_names, training_shape, censors).keys())
            == set(label_names)
        ), "Invalid keys in model censors"
        assert np.all(
            np.all(
                [
                    m.censors[k] == Censors(label_names, training_shape, censors)[k]
                    for k in label_names
                ]
            )
        ), "Bad censoring array"

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_censoring_censor_input(
        self, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        censors = Censors(
            label_names,
            training_shape,
            {l: np.zeros(training_shape, dtype=bool) for l in label_names},
        )

        m = model.CannonModel(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            censors=censors,
        )
        assert (
            set(m.censors.keys()) == censors.keys() == set(label_names)
        ), f"Invalid keys in model censors: {set(m.censors.keys())} vs {censors.keys()} vs {set(label_names)}"
        assert np.all(
            np.all([m.censors[k] == censors[k] for k in label_names])
        ), "Bad censoring array"
