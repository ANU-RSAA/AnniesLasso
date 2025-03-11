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

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("input_type", ["recarray", "ndarray"])
    def test_cannonmodel_training_set_labels(
        self, vectorizer, label_names, terms, training_shape, input_type
    ):
        """
        This test uses np.recarray as a proxy for all table-like inputs this class __init__ could accept.
        """
        if training_shape is None:
            fluxes = None
            ivar = None
        else:
            fluxes = np.ones((training_shape, 1))
            ivar = np.ones((training_shape, 1))
        vec = vectorizer(label_names=label_names, terms=terms)
        label_shape = training_shape if training_shape is not None else 10

        if input_type == "recarray":
            training_labels = np.recarray(
                (label_shape,), names=label_names, formats=["f8" for _ in label_names]
            )
            for i, k in enumerate(label_names, start=1):
                training_labels[k] = (
                    np.ones(label_shape) * i
                )  # Use integer to track values
        elif input_type == "ndarray":
            training_labels = np.zeros((label_shape, len(label_names)))
            for i, _ in enumerate(label_names, start=1):
                training_labels[:, i - 1] = np.ones(label_shape) * i

        m = model.CannonModel(training_labels, fluxes, ivar, vec)

        assert (
            type(m.training_set_labels) == np.ndarray
        ), "training labels table not converted to correct type"
        assert m.training_set_labels.shape == (
            label_shape,
            len(label_names),
        ), "training labels table converted to wrong shape"
        for i, k in enumerate(label_names, start=1):
            assert np.all(
                m.training_set_labels[:, i - 1] == np.ones(label_shape) * i
            ), "Training set labels not assigned to right label column!"

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    def test_cannonmodel_training_labels_missing_keys(
        self, vectorizer, label_names, terms, training_shape
    ):
        if training_shape is None:
            fluxes = None
            ivar = None
        else:
            fluxes = np.ones((training_shape, 1))
            ivar = np.ones((training_shape, 1))
        vec = vectorizer(label_names=label_names, terms=terms)
        label_shape = training_shape if training_shape is not None else 10

        training_labels = np.recarray(
                (label_shape,), names=["y", "z"], formats=["f8", "f8"]
        )

        with pytest.raises(ValueError, match="Unable to rectify"):
            m = model.CannonModel(training_labels, fluxes, ivar, vec)

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("labels_set_shape", [
        (50, 2),  # Will never match flux, may match terms
        (100, 4), # May match flux, will never match terms
        ])
    def test_cannonmodel_training_labels_bad_size(
        self, vectorizer, label_names, terms, training_shape, labels_set_shape
    ):
        # This actually works for the (50, 2) and terms 2 case
        if len(label_names) == 2 and labels_set_shape == (50, 2):
            pytest.skip()

        if training_shape is None:
            fluxes = None
            ivar = None
        else:
            fluxes = np.ones((training_shape, 1))
            ivar = np.ones((training_shape, 1))
        vec = vectorizer(label_names=label_names, terms=terms)

        training_labels = np.array(np.ones(labels_set_shape))

        with pytest.raises(ValueError):
            _ = model.CannonModel(training_labels, fluxes, ivar, vec)

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

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("no_of_stars", [1, 10, 100, 1000])
    @pytest.mark.parametrize("trained", [True, False])
    @mock.patch("thecannon.model.CannonModel.is_trained", new_callable=mock.PropertyMock)
    def test_cannonmodel_str(
        self, mock_is_trained, vectorizer, label_names, terms, training_shape, no_of_stars, trained
    ):
        if training_shape is None:
            fluxes = None
            ivar = None
        else:
            fluxes = np.ones((no_of_stars, training_shape))
            ivar = np.ones((no_of_stars, training_shape))
        training_set_labels = np.ones((no_of_stars, len(label_names)))
        vec = vectorizer(label_names=label_names, terms=terms)
        mock_is_trained.return_value = trained

        m = model.CannonModel(training_set_labels, fluxes, ivar, vec)
        str_rep = str(m)

        assert ("trained" in str_rep) == trained, "String rep wrong on training status"
        assert f"{len(label_names)} labels" in str_rep, "Missing/wrong number of labels in string rep"
        if training_shape is None:
            assert "no pixels" in str_rep, "Str rep not showing lack of training pixels"
        else:
            assert f"{training_shape} pixels" in str_rep, "Str rep not showing right no of pixels"
        assert f"{no_of_stars} stars" in str_rep, "Str rep not showing right no of stars"

    def test_cannonmodel_repr(
            self, vectorizer, label_names, terms
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        m = model.CannonModel(np.ones((10, len(label_names))), None, None, vec)

        str_rep = m.__repr__()
        assert "model" in str_rep, "Didn't get correct module name"
        assert "CannonModel" in str_rep, "Didn't get correct class name"
    @pytest.mark.parametrize("test_value", [
        "Test value",
        1,
        1.0,
        np.ones(10),
        np.zeros((10, 10)),
    ])
    class TestCannonModelPropertyRetrieval:
        def test_cannonmodel_theta_property(self, vectorizer, label_names, terms, test_value):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = model.CannonModel(np.ones((10, len(label_names))), None, None, vec)

            m._theta = test_value
            assert np.all(m.theta == test_value), "Theta property broken"

        def test_cannonmodel_s2_property(self, vectorizer, label_names, terms, test_value):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = model.CannonModel(np.ones((10, len(label_names))), None, None, vec)

            m._s2 = test_value
            assert np.all(m.s2 == test_value), "s2 property broken"

        def test_cannonmodel_design_matrix_property(self, vectorizer, label_names, terms, test_value):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = model.CannonModel(np.ones((10, len(label_names))), None, None, vec)

            m._design_matrix = test_value
            assert np.all(m.design_matrix == test_value), "design_matrix property broken"
    
