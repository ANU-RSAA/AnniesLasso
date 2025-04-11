#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from thecannon import model, restricted
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
    "input",
    [
        1.0,
        [0.0, 1.0, 2.0],
    ],
)
def test__pixel_access(input):
    assert (
        model.CannonModel._pixel_access(input, 1, default=None) == 1.0
    ), "_pixel_access not behaving as expected"


@pytest.mark.parametrize(
    "input",
    [
        np.ones(10),
        np.ones(100),
    ],
)
@pytest.mark.parametrize("offset", [1, 3, 10, 100])
@pytest.mark.parametrize("default", [0.0, 3.0])
def test__pixel_access_out_of_bounds(input, offset, default):
    assert (
        model.CannonModel._pixel_access(input, len(input) + offset, default=default)
        == default
    ), "Unexpected out of bounds behaviour"


@pytest.mark.parametrize("default", [0.0, 3.0])
@pytest.mark.parametrize("index", [0, 1, 10])
def test__pixel_access_input_none(default, index):
    assert (
        model.CannonModel._pixel_access(None, index, default=default) == default
    ), "Unexpected behaviour for None input array"


@pytest.mark.parametrize("test_model,module,name", [
    (model.CannonModel, "model", "CannonModel"),
    (restricted.RestrictedCannonModel, "restricted", "RestrictedCannonModel"),
])
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
        (
            ["Teff", "H", "z"],
            [[("Teff", 1)], [("Teff", 2), ("H", 1)], [("H", 2), ("z", 3)]],
        ),
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
        test_model,
        module,
        name,
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
            _ = test_model(
                training_set_labels, training_set_flux, training_set_ivar, vec
            )

    @pytest.mark.parametrize(
        "training_set_shape", [(10,), (10, 10), (100,), (100, 100), (10, 100)]
    )
    @pytest.mark.parametrize("inf_posn", [0, 25, 50, 75, 99])
    class TestCannonModelInfValues(object):
        def test_cannonmodel_inf_flux(
            self, test_model, module, name, vectorizer, label_names, terms, training_set_shape, inf_posn
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            training_set_flux = np.ones(training_set_shape)
            training_set_flux[
                int((inf_posn / 100.0) * training_set_flux.shape[0])
            ] = np.inf
            training_set_ivar = np.ones(training_set_shape)
            training_set_labels = np.ones((training_set_shape[0], len(label_names)))

            with pytest.raises(ValueError, match="fluxes.*finite"):
                _ = test_model(
                    training_set_labels, training_set_flux, training_set_ivar, vec
                )

        def test_cannonmodel_inf_ivar(
            self, test_model, module, name, vectorizer, label_names, terms, training_set_shape, inf_posn
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            training_set_flux = np.ones(training_set_shape)
            training_set_ivar = np.ones(training_set_shape)
            training_set_ivar[
                int((inf_posn / 100.0) * training_set_ivar.shape[0])
            ] = np.inf
            training_set_labels = np.ones(training_set_shape)

            with pytest.raises(ValueError, match="ivars.*finite"):
                _ = test_model(
                    training_set_labels, training_set_flux, training_set_ivar, vec
                )

    def test_cannonmodel_no_training_set_labels(self, test_model, module, name, vectorizer, label_names, terms):
        vec = vectorizer(label_names=label_names, terms=terms)
        with pytest.raises(ValueError):
            _ = test_model(None, None, None, vec)

    @pytest.mark.parametrize("training_labels_length", [10, 100, 1000])
    def test_cannonmodel_blank_flux_and_ivar(
        self, test_model, module, name, vectorizer, label_names, terms, training_labels_length
    ):
        training_labels = np.ones((training_labels_length, len(label_names)))
        vec = vectorizer(label_names=label_names, terms=terms)
        m = test_model(training_labels, None, None, vec)
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
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, input_type
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

        m = test_model(training_labels, fluxes, ivar, vec)

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

        m2 = test_model(training_labels, fluxes, ivar, vec)
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    @pytest.mark.parametrize("input_type", ["recarray", "ndarray"])
    def test_cannonmodel_training_set_labels_bad_value(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, bad_value, input_type
    ):
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
                training_labels[k] = np.ones(label_shape) * i
                training_labels[k][0] = bad_value  # Use integer to track values
        elif input_type == "ndarray":
            training_labels = np.zeros((label_shape, len(label_names)))
            for i, _ in enumerate(label_names, start=1):
                training_labels[:, i - 1] = np.ones(label_shape) * i
                training_labels[:, 0] = bad_value

        with pytest.raises(ValueError, match="not all finite"):
            m = test_model(training_labels, fluxes, ivar, vec)

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    def test_cannonmodel_training_labels_missing_keys(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
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
            m = test_model(training_labels, fluxes, ivar, vec)

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize(
        "labels_set_shape",
        [
            (50, 2),  # Will never match flux, may match terms
            (100, 4),  # May match flux, will never match terms
        ],
    )
    def test_cannonmodel_training_labels_bad_size(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, labels_set_shape
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
            _ = test_model(training_labels, fluxes, ivar, vec)

    @pytest.mark.parametrize("correlation_direction", [1.1, -1.1])
    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_training_labels_correlated_warning(
        self,
        test_model,
        module,
        name,
        vectorizer,
        label_names,
        terms,
        correlation_direction,
        training_shape,
        caplog,
    ):
        training_labels = np.random.random((training_shape, len(label_names)))
        # Make the first term monotonically increase
        training_labels[:, 0] = np.asarray(
            [float(i) for i in range(training_labels.shape[0])]
        )
        # Make the last term monotonically increase/decrease according to correlation_direction
        training_labels[:, -1] = training_labels[:, 0] * correlation_direction
        # Leave all other terms random (so hopefully, totally uncorrelated)

        m = test_model(
            training_labels,
            None,
            None,
            vectorizer(label_names=label_names, terms=terms),
        )

        if len(label_names) == 1:  # Should be no messages about correlation
            assert (
                "are highly correlated" not in caplog.text
            ), "Received a correlation warning with only one label!"
        else:  # Should have received a correlation warning (+/-) with first and last labels
            assert (
                f"Labels '{label_names[-1]}' and '{label_names[0]}' are highly correlated"
                in caplog.text
            ), "Did not get expected correlation warning!"

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_censoring_dict_input(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        censors = {l: np.zeros((training_shape,), dtype=bool) for l in label_names}

        m = test_model(
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

        m2 = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            censors=censors,
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_censoring_censor_input(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
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

        m = test_model(
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

        m2 = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            censors=censors,
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_censoring_censor_input_bad_vec_terms(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        censors = Censors(
            label_names + ["z"],
            training_shape,
            {l: np.zeros(training_shape, dtype=bool) for l in label_names},
        )

        with pytest.raises(ValueError, match="Censor label names !="):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                censors=censors,
            )

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_censoring_censor_input_bad_num_pixels(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        censors = Censors(
            label_names,
            training_shape + 1,
            {l: np.zeros(training_shape + 1, dtype=bool) for l in label_names},
        )

        with pytest.raises(ValueError, match="Censor num_pixels !="):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                censors=censors,
            )

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    @pytest.mark.parametrize("censors", [list(), 1, 1.0, "dict", "censors"])
    def test_cannonmodel_censoring_censor_input_bad_type(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, censors
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        with pytest.raises(TypeError, match="censors must be"):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                censors=censors,
            )

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("no_of_stars", [1, 10, 100, 1000])
    @pytest.mark.parametrize("trained", [True, False])
    @mock.patch(
        "thecannon.model.CannonModel.is_trained", new_callable=mock.PropertyMock
    )
    def test_cannonmodel_str(
        self,
        mock_is_trained,
        test_model,
        module,
        name,
        vectorizer,
        label_names,
        terms,
        training_shape,
        no_of_stars,
        trained,
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

        m = test_model(training_set_labels, fluxes, ivar, vec)
        str_rep = str(m)

        assert ("trained" in str_rep) == trained, "String rep wrong on training status"
        assert (
            f"{len(label_names)} labels" in str_rep
        ), "Missing/wrong number of labels in string rep"
        if training_shape is None:
            assert "no pixels" in str_rep, "Str rep not showing lack of training pixels"
        else:
            assert (
                f"{training_shape} pixels" in str_rep
            ), "Str rep not showing right no of pixels"
        assert (
            f"{no_of_stars} stars" in str_rep
        ), "Str rep not showing right no of stars"

    def test_cannonmodel_repr(self, test_model, module, name, vectorizer, label_names, terms):
        vec = vectorizer(label_names=label_names, terms=terms)
        m = test_model(np.ones((10, len(label_names))), None, None, vec)

        str_rep = m.__repr__()
        assert module in str_rep, "Didn't get correct module name"
        assert name in str_rep, "Didn't get correct class name"

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    def test_cannonmodel_dispersion_input(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            training_set_flux = None
            training_set_ivar = None
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        dispersion = np.ones(training_shape if training_shape is not None else 10)

        m = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            dispersion=dispersion,
        )

        assert np.all(
            m.dispersion == dispersion
        ), "Dispersion not correctly carried through"

        m2 = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            dispersion=dispersion,
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    def test_cannonmodel_dispersion_input_bad_size(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            pytest.skip()
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        dispersion = np.ones(training_shape + training_shape)

        with pytest.raises(
            ValueError, match="does not match the number of pixels per star"
        ):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                dispersion=dispersion,
            )

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("bad_value", [np.inf, np.nan])
    def test_cannonmodel_dispersion_input_bad_value(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, bad_value
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            training_set_flux = None
            training_set_ivar = None
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        dispersion = np.ones(training_shape if training_shape is not None else 10)
        dispersion[-1] = bad_value
        with pytest.raises(ValueError, match="must be finite"):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                dispersion=dispersion,
            )

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("bad_type", [str, np.string_])
    def test_cannonmodel_dispersion_input_bad_type(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, bad_type
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            training_set_flux = None
            training_set_ivar = None
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        dispersion = np.ones(
            training_shape if training_shape is not None else 10, dtype=bad_type
        )
        with pytest.raises(ValueError, match="are not float-like"):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                dispersion=dispersion,
            )

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize(
        "regularization, reg_expected",
        [
            (1.0, 1.0),
            (
                [
                    1.0,
                ],
                1.0,
            ),
            (None, None),
        ],
    )
    def test_cannonmodel_regularization_input(
        self,
        test_model,
        module,
        name,
        vectorizer,
        label_names,
        terms,
        training_shape,
        regularization,
        reg_expected,
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            training_set_flux = None
            training_set_ivar = None
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        if regularization is None:
            regularization = np.ones(
                training_shape if training_shape is not None else 10
            )

        m = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            regularization=regularization,
        )

        assert np.all(
            m.regularization
            == (reg_expected if reg_expected is not None else regularization)
        ), "Unexpected regularization set"

        m2 = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            regularization=regularization,
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize("training_shape", [10, 100, 1000])
    def test_cannonmodel_regularization_bad_shape(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        training_set_flux = np.ones((1, training_shape))
        training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        regularization = np.ones(training_shape + training_shape)

        with pytest.raises(ValueError, match="must be of size `num_pixels`"):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                regularization=regularization,
            )

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize(
        "regularization",
        [
            1.0,
            [
                1.0,
            ],
            None,
        ],
    )
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1, -1.0, "a"])
    def test_cannonmodel_regularization_input_bad_value(
        self, test_model, module, name, vectorizer, label_names, terms, training_shape, regularization, bad_value
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            training_set_flux = None
            training_set_ivar = None
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        if regularization is None:
            regularization = np.ones(
                training_shape if training_shape is not None else 10,
                dtype=type(bad_value),
            )

        if isinstance(regularization, np.ndarray):
            regularization[0] = bad_value
        else:
            regularization = bad_value

        with pytest.raises(ValueError, match="must be positive and finite"):
            m = test_model(
                training_set_labels,
                training_set_flux,
                training_set_ivar,
                vec,
                regularization=regularization,
            )

    @pytest.mark.parametrize("training_shape", [None, 10, 100, 1000])
    @pytest.mark.parametrize("censors", [None, dict()])
    @pytest.mark.parametrize(
        "dispersion",
        [
            None,
            [
                1.0,
            ],
        ],
    )
    @pytest.mark.parametrize(
        "regularization",
        [
            None,
            1.0,
            [
                1.0,
            ],
        ],
    )
    def test_cannonmodel_training_status(
        self,
        test_model,
        module,
        name,
        vectorizer,
        label_names,
        terms,
        training_shape,
        censors,
        dispersion,
        regularization,
    ):
        vec = vectorizer(label_names=label_names, terms=terms)
        if training_shape is None:
            training_set_flux = None
            training_set_ivar = None
        else:
            training_set_flux = np.ones((1, training_shape))
            training_set_ivar = np.ones((1, training_shape))
        training_set_labels = np.ones((1, len(label_names)))

        if censors is not None:
            censors = {
                l: np.ones(training_shape if training_shape is not None else 10)
                for l in label_names
            }

        if type(regularization) == list:
            regularization = regularization * (
                training_shape if training_shape is not None else 10
            )

        if type(dispersion) == list:
            dispersion = dispersion * (
                training_shape if training_shape is not None else 10
            )

        m = test_model(
            training_set_labels,
            training_set_flux,
            training_set_ivar,
            vec,
            regularization=regularization,
            censors=censors,
            dispersion=dispersion,
        )

        assert m.is_trained == False, "Model incorrectly marked trained after init!"

        # Mock the trained attributes
        for attr in m._trained_attributes:
            setattr(m, f"_{attr}", 1.0)

        assert m.is_trained, "Model not reporting trained as required!"

        m.reset()
        assert not m.is_trained, "Model reporting trained after reset!"

    @pytest.mark.parametrize(
        "scale_labels_function",
        [
            lambda l: np.ptp(
                np.percentile(l, [2.5, 97.5], axis=0), axis=0, keepdims=False
            ),
            lambda l: np.ptp(
                np.percentile(l, [10.0, 75.0], axis=0), axis=0, keepdims=False
            ),
        ],
    )
    def test_cannonmodel__scales_init(
        self, test_model, module, name, vectorizer, label_names, terms, scale_labels_function
    ):
        test_labels = np.ones((10, len(label_names)))
        for i in range(10):
            test_labels[i, :] = i
        vec = vectorizer(label_names=label_names, terms=terms)
        m = test_model(
            test_labels, None, None, vec, __scale_labels_function=scale_labels_function
        )

        assert np.all(m._scales == scale_labels_function(test_labels))

        m2 = test_model(
            test_labels, None, None, vec, __scale_labels_function=scale_labels_function
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize(
        "scale_labels_function",
        [
            lambda l: np.ptp(
                np.percentile(l, [2.5, 97.5], axis=0), axis=0, keepdims=True
            ),
            lambda l: np.ptp(
                np.percentile(l, [10.0, 75.0], axis=0), axis=0, keepdims=True
            ),
            lambda l: np.mean(l),
            np.mean,  # This method of input will actually work, if given the right output shape - this isn't though
            "a",
            3,
            [1, 2, 3],
            "np.ptp",
        ],
    )
    def test_cannonmodel__scales_init_bad(
        self, test_model, module, name, vectorizer, label_names, terms, scale_labels_function
    ):
        test_labels = np.ones((10, len(label_names)))
        for i in range(10):
            test_labels[i, :] = i
        vec = vectorizer(label_names=label_names, terms=terms)
        with pytest.raises(ValueError):
            m = test_model(
                test_labels,
                None,
                None,
                vec,
                __scale_labels_function=scale_labels_function,
            )

    @pytest.mark.parametrize(
        "fiducial_labels_function",
        [
            lambda l: np.percentile(l, 65, axis=0),
            lambda l: np.percentile(l, 10, axis=0),
            lambda l: np.mean(l, axis=0),
            lambda l: np.median(l, axis=0),
        ],
    )
    def test_cannonmodel__fiducials_init(
        self, test_model, module, name, vectorizer, label_names, terms, fiducial_labels_function
    ):
        test_labels = np.ones((10, len(label_names)))
        for i in range(10):
            test_labels[i, :] = i
        vec = vectorizer(label_names=label_names, terms=terms)
        m = test_model(
            test_labels,
            None,
            None,
            vec,
            __fiducial_labels_function=fiducial_labels_function,
        )

        assert np.all(m._fiducials == fiducial_labels_function(test_labels))

        m2 = test_model(
            test_labels,
            None,
            None,
            vec,
            __fiducial_labels_function=fiducial_labels_function,
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    @pytest.mark.parametrize(
        "fiducial_labels_function",
        [
            lambda l: np.ptp(
                np.percentile(l, [2.5, 97.5], axis=0), axis=0, keepdims=True
            ),
            lambda l: np.ptp(
                np.percentile(l, [10.0, 75.0], axis=0), axis=0, keepdims=True
            ),
            lambda l: np.mean(l),
            np.mean,  # This method of input will actually work, if given the right output shape - this isn't though
            "a",
            3,
            [1, 2, 3],
            "np.ptp",
        ],
    )
    def test_cannonmodel__fiducials_init_bad(
        self, test_model, module, name, vectorizer, label_names, terms, fiducial_labels_function
    ):
        test_labels = np.ones((10, len(label_names)))
        for i in range(10):
            test_labels[i, :] = i
        vec = vectorizer(label_names=label_names, terms=terms)
        with pytest.raises(ValueError):
            m = test_model(
                test_labels,
                None,
                None,
                vec,
                __fiducial_labels_function=fiducial_labels_function,
            )

    @pytest.mark.parametrize(
        "scale_labels_function",
        [
            lambda l: np.ptp(
                np.percentile(l, [2.5, 97.5], axis=0), axis=0, keepdims=False
            ),
            lambda l: np.ptp(
                np.percentile(l, [10.0, 75.0], axis=0), axis=0, keepdims=False
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fiducial_labels_function",
        [
            lambda l: np.percentile(l, 65, axis=0),
            lambda l: np.percentile(l, 10, axis=0),
            lambda l: np.mean(l, axis=0),
            lambda l: np.median(l, axis=0),
        ],
    )
    def test_cannonmodel__design_matrix_init(
        self,
        test_model,
        module,
        name,
        vectorizer,
        label_names,
        terms,
        scale_labels_function,
        fiducial_labels_function,
    ):
        test_labels = np.ones((10, len(label_names)))
        for i in range(10):
            test_labels[i, :] = i
        vec = vectorizer(label_names=label_names, terms=terms)
        m = test_model(
            test_labels,
            None,
            None,
            vec,
            __scale_labels_function=scale_labels_function,
            __fiducial_labels_function=fiducial_labels_function,
        )

        assert np.all(
            m._design_matrix
            == vec((m.training_set_labels - m._fiducials) / m._scales).T
        ), "definition of _design_matrix has changed!"

        m2 = test_model(
            test_labels,
            None,
            None,
            vec,
            __scale_labels_function=scale_labels_function,
            __fiducial_labels_function=fiducial_labels_function,
        )
        assert m == m2, "__eq__ does not recognize identically-created Model"

    # FIXME work out reliable test training_set_labels
    @pytest.mark.skip
    @pytest.mark.parametrize(
        "out_of_hull_indices",
        [
            [(2, -1)],
            [(4, -1), (3, -1), (2, -1)],
        ],
    )
    def test_cannonmodel_in_convex_hull(
        self, test_model, module, name, vectorizer, label_names, terms, out_of_hull_indices
    ):
        training_labels = np.random.random(
            (10, len(label_names))
        )

        m = test_model(
            training_labels,
            None,
            None,
            vectorizer(label_names=label_names, terms=terms),
        )

        test_labels = np.ones((100, len(label_names))) * 0.5  ## All inside hull

        if len(label_names) == 1:
            with pytest.raises(RuntimeError, match="with a single label"):
                _ = m.in_convex_hull(test_labels)

        else:
            assert np.all(
                m.in_convex_hull(test_labels)
            ), "All test_labels should be within hull at first"

            for ij in out_of_hull_indices:
                test_labels[ij] = 2.0  # Outside hull

            ich = m.in_convex_hull(test_labels)
            assert ~np.all(
                ich
            ), "There should now be some labels outside the convex hull"
            assert np.count_nonzero(~ich) == len(
                out_of_hull_indices
            ), f"Should have {len(out_of_hull_indices)} out of hull points, have {np.count_nonzero(ich)}"

    @pytest.mark.parametrize("bad_label_size", [6, 10])
    def test_cannonmodel_in_convex_hull_bad_label_size(
        self, test_model, module, name, vectorizer, label_names, terms, bad_label_size
    ):
        if len(label_names) == 1:
            pytest.skip()

        training_labels = np.random.random(
            (10, len(label_names))
        )  # Guaranteed to be in range (0, 1)

        m = test_model(
            training_labels,
            None,
            None,
            vectorizer(label_names=label_names, terms=terms),
        )

        test_labels = np.ones((100, bad_label_size)) * 0.5  ## All inside hull

        with pytest.raises(ValueError, match="expected"):
            _ = m.in_convex_hull(test_labels)

    @pytest.mark.parametrize("training_set_shape", [
        3, 9, 22
    ])
    class TestCannonModelNeTrainingDiffs:
        def test_cannonmodel_ne_training_set_flux(self, test_model, module, name, vectorizer, label_names, terms, training_set_shape):
            training_set_flux = np.ones((training_set_shape, len(label_names)))
            training_set_ivar = np.ones((training_set_shape, len(label_names)))
            training_set_labels = np.ones((training_set_shape, len(label_names)))

            m1 = test_model(training_set_labels, training_set_flux, training_set_ivar, vectorizer(label_names=label_names, terms=terms))
            m2 = test_model(training_set_labels, 
                            np.zeros((training_set_shape, len(label_names))), 
                            training_set_ivar, vectorizer(label_names=label_names, terms=terms))
            
            assert m1 != m2, "__eq__ failed to detect differing training_set_flux"

        def test_cannonmodel_ne_training_set_ivar(self, test_model, module, name, vectorizer, label_names, terms, training_set_shape):
            training_set_flux = np.ones((training_set_shape, len(label_names)))
            training_set_ivar = np.ones((training_set_shape, len(label_names)))
            training_set_labels = np.ones((training_set_shape, len(label_names)))

            m1 = test_model(training_set_labels, training_set_flux, training_set_ivar, vectorizer(label_names=label_names, terms=terms))
            m2 = test_model(training_set_labels, 
                            training_set_flux, 
                            np.zeros((training_set_shape, len(label_names))), vectorizer(label_names=label_names, terms=terms))
            
            assert m1 != m2, "__eq__ failed to detect differing training_set_ivar"

        def test_cannonmodel_ne_training_set_labels(self, test_model, module, name, vectorizer, label_names, terms, training_set_shape):
            training_set_flux = np.ones((training_set_shape, len(label_names)))
            training_set_ivar = np.ones((training_set_shape, len(label_names)))
            training_set_labels = np.ones((training_set_shape, len(label_names)))

            m1 = test_model(training_set_labels, training_set_flux, training_set_ivar, vectorizer(label_names=label_names, terms=terms))
            m2 = test_model(np.zeros((training_set_shape, len(label_names))), 
                            training_set_flux, 
                            training_set_ivar, vectorizer(label_names=label_names, terms=terms))
            
            assert m1 != m2, "__eq__ failed to detect differing training_labels"

        def test_cannonmodel_ne_training_set_None(self, test_model, module, name, vectorizer, label_names, terms, training_set_shape):
            training_set_flux = np.ones((training_set_shape, len(label_names)))
            training_set_ivar = np.ones((training_set_shape, len(label_names)))
            training_set_labels = np.ones((training_set_shape, len(label_names)))

            m1 = test_model(training_set_labels, training_set_flux, training_set_ivar, vectorizer(label_names=label_names, terms=terms))
            m2 = test_model(training_set_labels, 
                            None, 
                            None, vectorizer(label_names=label_names, terms=terms))

    @pytest.mark.parametrize(
        "test_value",
        [
            "Test value",
            1,
            1.0,
            np.ones(10),
            np.zeros((10, 10)),
        ],
    )
    class TestCannonModelPropertyRetrieval:
        """
        These tests are designed to ensure the property getter function retrieves exactly
        what is in the hidden attribute without modification - some of these test data types
        will be invalid inputs to the setter
        """

        def test_cannonmodel_theta_property(
            self, test_model, module, name, vectorizer, label_names, terms, test_value
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = test_model(np.ones((10, len(label_names))), None, None, vec)

            m._theta = test_value
            assert np.all(m.theta == test_value), "Theta property broken"

        def test_cannonmodel_s2_property(
            self, test_model, module, name, vectorizer, label_names, terms, test_value
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = test_model(np.ones((10, len(label_names))), None, None, vec)

            m._s2 = test_value
            assert np.all(m.s2 == test_value), "s2 property broken"

        def test_cannonmodel_design_matrix_property(
            self, test_model, module, name, vectorizer, label_names, terms, test_value
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = test_model(np.ones((10, len(label_names))), None, None, vec)

            m._design_matrix = test_value
            assert np.all(
                m.design_matrix == test_value
            ), "design_matrix property broken"

        def test_cannonmodel_dispersion_property(
            self, test_model, module, name, vectorizer, label_names, terms, test_value
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = test_model(np.ones((10, len(label_names))), None, None, vec)

            m._dispersion = test_value
            assert np.all(m.dispersion == test_value), "design_matrix property broken"

        def test_cannonmodel_regularization_property(
            self, test_model, module, name, vectorizer, label_names, terms, test_value
        ):
            vec = vectorizer(label_names=label_names, terms=terms)
            m = test_model(np.ones((10, len(label_names))), None, None, vec)

            m._regularization = test_value
            assert np.all(
                m.regularization == test_value
            ), "design_matrix property broken"
