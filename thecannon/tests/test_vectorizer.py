"""
Unit tests for `thecannon.vectorizer`.
"""

import pytest
from unittest import mock
import numpy as np

from ..vectorizer.base import BaseVectorizer
from ..vectorizer.polynomial import (
    PolynomialVectorizer,
    terminator,
    human_readable_label_vector,
    human_readable_label_term,
    _is_structured_label_vector,
    parse_label_vector_description,
)


@pytest.mark.parametrize("vectorizer", [BaseVectorizer, PolynomialVectorizer])
class TestVectorizersCommon:
    @pytest.mark.parametrize(
        "label_names,terms,terms_out",
        [
            [
                ("a", "b", "c"),
                [[("a", 4)], [("b", 1)], [("c", 2)]],
                [[(0, 4)], [(1, 1)], [(2, 2)]],
            ],
            [["a", "b", "c"], [[(0, 4)], [(1, 1)], [(2, 2)]], None],
            (["Teff", "g"], [[(0, 1)], [(0, 2), (1, 1)], [(1, 2), (0, 2)]], None),
        ],
    )
    def test_vectorizer_basic_init(self, vectorizer, label_names, terms, terms_out):
        vec = vectorizer(label_names=label_names, terms=terms)
        assert vec.label_names == tuple(
            label_names
        ), "Label names not initialized correctly"
        assert vec.terms == (
            terms_out
            if (terms_out is not None and isinstance(vec, PolynomialVectorizer))
            else terms
        ), "Terms not initialized correctly"

    @pytest.mark.parametrize(
        "label_names,terms",
        [
            [("a", "b"), [[("b", 1)]]],  # Unused term
            [
                ("a",),
                [[("a", 1), ("b", 1), ("c", 1)]],
            ],  # Excess term (term number mismatch)
            [("a", "b"), [[("c", 1), ("a", 1)]]],  # Invalid term (str)
            [("a", "b"), [[(2, 1), (0, 1)]]],  # Invalid term (int)
            [("a", "b"), [[(1, 1), (0, 0)]]],  # Invalid power (int terms)
            [("a", "b"), [[("b", 1), ("a", 0)]]],  # Invalid power (str terms)
        ],
    )
    def test_vectorizer_basic_init_bad_inputs(self, vectorizer, label_names, terms):
        with pytest.raises(ValueError):
            vec = vectorizer(label_names=label_names, terms=terms)

    def test_vectorizer_no_direct_sets(self, vectorizer):
        vec = vectorizer(label_names=("a"), terms=[[("a", 1)]])  # Basic
        with pytest.raises(RuntimeError):
            vec.label_names = ("a", "b")
        with pytest.raises(RuntimeError):
            vec.terms = [[("a", 1), ("a", 2)]]

    def test_vectorizer__str__(self, vectorizer):
        vec = vectorizer(label_names=("a"), terms=[[("a", 1)]])
        str_rep = vec.__str__()
        assert str(vec) == str_rep, "String representation mismatch!"
        assert isinstance(str_rep, str), "__str__ did not return a string!"
        assert "1 labels" in str_rep, "Labels number not in __str__"
        assert "1 terms" in str_rep, "Terms number not in __str__"

    def test_vectorizer__repr__(self, vectorizer):
        vec = vectorizer(label_names=("a"), terms=[[("a", 1)]])
        repr_str = vec.__repr__()
        assert isinstance(repr_str, str), "__repr__ did not return a string!"

    def test_vectorizer__getstate__(self, vectorizer):
        vec = vectorizer(label_names=("a"), terms=[[("a", 1)]])
        state = vec.__getstate__()
        assert state[0] == type(vec).__name__, "Name not first element of state return!"
        assert isinstance(state[1], dict), "Dict not second element of state return!"
        assert (
            "label_names" in state[1].keys()
        ), "Could not ID label_names in state return!"
        assert "terms" in state[1].keys(), "Could not ID terms in state return!"
        assert "metadata" in state[1].keys(), "Could not ID metadata in state return!"

    def test_vectorizer__setstate__(self, vectorizer):
        vec = vectorizer(label_names=("a"), terms=[[("a", 1)]])
        blank_vec = vectorizer(label_names=("b"), terms=[[("b", 9)]])
        blank_vec.__setstate__(vec.__getstate__())
        assert (
            str(vec) == str(blank_vec)
            and vec.terms == blank_vec.terms
            and vec.label_names == blank_vec.label_names
            and vec.metadata == blank_vec.metadata
        ), "__setstate__ failed to copy across vectorizer status!"


def test_base_vectorizer__call__():
    vec = BaseVectorizer(label_names=None, terms=None)
    with pytest.raises(NotImplementedError):
        vec.__call__(None)


def test_base_vectorizer_get_label_vector():
    vec = BaseVectorizer(label_names=None, terms=None)
    with pytest.raises(NotImplementedError):
        vec.get_label_vector(None)


def test_base_vectorizer_get_label_vector_derivative():
    vec = BaseVectorizer(label_names=None, terms=None)
    with pytest.raises(NotImplementedError):
        vec.get_label_vector_derivative(None)


@pytest.mark.parametrize(
    "label_names,terms,terms_out,order",
    [
        [("a", "b", "c"), "a^3 + b + c^2", [[(0, 3)], [(1, 1)], [(2, 2)]], None],
        (
            ["Teff", "g"],
            "Teff + Teff^2*g + g^2*Teff^2",
            [[(0, 1)], [(0, 2), (1, 1)], [(1, 2), (0, 2)]],
            None,
        ),
    ],
)
def test_polynomial_vectorizer_basic_init(label_names, terms, terms_out, order):
    vec = PolynomialVectorizer(label_names=label_names, order=order, terms=terms)
    assert vec.label_names == tuple(
        label_names
    ), "Label names not initialized correctly"
    assert vec.terms == (
        terms_out if terms_out is not None else terms
    ), "Terms not initialized correctly"


@pytest.mark.parametrize(
    "bad_labels",
    [
        [
            [
                [
                    "a",
                ]
            ]
        ],  # 3-dim
        [
            [
                [
                    [
                        "a",
                    ]
                ]
            ]
        ],  # 4-dim
    ],
)
def test_polynomial_vectorizer_get_label_vector_bad_input(bad_labels):
    vec = PolynomialVectorizer(label_names=["a"], order=1)
    with pytest.raises(ValueError):
        vec.get_label_vector(bad_labels)


@pytest.mark.parametrize(
    "label_names,order,cross_term_order,expected",
    [
        (["a"], 4, -1, "a + a^2 + a^3 + a^4"),
        (["a", "b"], 2, -1, "a + b + a^2 + a*b + b^2"),
        (["a", "b", "c"], 0, -1, ""),
        (["a", "b", "c"], -1, -1, ""),
        (
            ["a", "b", "c"],
            1,
            3,
            "a + b + c + a*b + a*c + b*c + a^2*b + a^2*c + a*b^2 + a*b*c + a*c^2 + b^2*c + b*c^2 + a^3*b + a^3*c + a^2*b^2 + a^2*b*c + a^2*c^2 + a*b^3 + a*b^2*c + a*b*c^2 + a*c^3 + b^3*c + b^2*c^2 + b*c^3",
        ),
    ],
)
def test_terminator(label_names, order, cross_term_order, expected):
    ret = terminator(label_names, order, cross_term_order=cross_term_order)
    assert (
        ret == expected
    ), f"Expected return {expected} does not match actual return {ret}"


@pytest.mark.parametrize(
    "sep,mul,pow",
    [
        ("s", "x", "p"),
    ],
)
def test_terminator_format_kwargs(sep, mul, pow):
    ret = terminator(["a", "b"], 2, -1, sep=sep, mul=mul, pow=pow)
    default_ret = terminator(["a", "b"], 2, -1)
    assert ret == default_ret.replace("+", sep).replace("*", mul).replace("^", pow)


@pytest.mark.parametrize(
    "bad_cross_term_order",
    [
        "a",
        19.2,
        "11",
        # -44.8  # This is actually OK - anything that succeeds the < 0 comparison will be set to order - 1
    ],
)
def test_terminator_bad_cross_term_order(bad_cross_term_order):
    with pytest.raises(TypeError):
        _ = terminator(["a", "b"], 2, bad_cross_term_order)


# Arguments here are garbage - deliberate to ensure error is thrown
# before they are looked at
@pytest.mark.parametrize(
    "label_names,order,terms",
    [
        (None, "a", None),  # terms is None and None in (label_names, order)
        ("a", None, None),  # terms is None and None in (label_names, order)
        ("a", "a", "a"),  # terms is not None and order is not None
        (None, "a", "a"),  # terms is not None and order is not None
    ],
)
def test_polynomial_vectorizer_argument_dichotomy(label_names, order, terms):
    with pytest.raises(ValueError):
        PolynomialVectorizer(label_names=label_names, terms=terms, order=order)


@pytest.mark.parametrize("aspect", ["_terms", "_label_names"])
def test_polynomial_vectorizer_index_labels_bad_vec(aspect):
    vec = PolynomialVectorizer(label_names=["a", "b"], order=2)
    with mock.patch.object(vec, aspect, None):
        with pytest.raises(ValueError):
            vec.index_labels()


@pytest.mark.parametrize(
    "label_names,order,terms,terms_indexed",
    [
        (
            ["a", "b"],
            2,
            [[("a", 1)], [("b", 1)], [("a", 2)], [("a", 1), ("b", 1)], [("b", 2)]],
            [[(0, 1)], [(1, 1)], [(0, 2)], [(0, 1), (1, 1)], [(1, 2)]],
        ),
        (
            ["a", "b"],
            2,
            "a + b + a^2 + a*b + b^2",
            [
                [
                    (0, 1),
                ],
                [
                    (1, 1),
                ],
                [
                    (0, 2),
                ],
                [(0, 1), (1, 1)],
                [
                    (1, 2),
                ],
            ],
        ),
    ],
)
class TestVectorizerInits:
    def test_polynomial_vectorizer_argument_equivalence(
        self, label_names, order, terms, terms_indexed
    ):
        vec1 = PolynomialVectorizer(label_names=label_names, order=order)
        vec2 = PolynomialVectorizer(label_names=label_names, terms=terms)
        vec3 = PolynomialVectorizer(terms=terms, label_names=None, order=None)

        assert str(vec1) == str(vec2) == str(vec3), "String comparison failed!"
        assert vec1.terms == vec2.terms == vec3.terms, "Terms comparison failed!"
        assert (
            vec1.label_names == vec2.label_names == vec3.label_names
        ), "Label names comparison failed"

    def test_polynomial_vectorizer_index_labels(
        self, label_names, order, terms, terms_indexed
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        vec.index_labels()
        assert vec.terms == terms_indexed, "index_labels did not work as expected!"

    def test_polynomial_vectorizer_get_label_vector_noterms(
        self, label_names, order, terms, terms_indexed
    ):
        vec = PolynomialVectorizer(order=order, label_names=label_names)
        with mock.patch.object(
            vec, "_terms", None
        ):  # Must patch underlying value, not getter
            with pytest.raises(RuntimeError):
                vec.get_label_vector(label_names)

    def test_polynomial_vectorizer_get_label_vector_1D(
        self, label_names, order, terms, terms_indexed
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        t = vec.get_label_vector(np.ones(len(vec.label_names)))

        assert t.shape == (
            len(vec.terms) + 1,
            1,
        ), "Unexpected label_vector output size (1D)"

    @pytest.mark.parametrize("N", [1, 3, 6, 10, 100])
    def test_polynomial_vectorizer_get_label_vector(
        self, label_names, order, terms, terms_indexed, N
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        t = vec.get_label_vector(np.ones((N, len(vec.label_names))))

        assert t.shape == (
            len(vec.terms) + 1,
            N,
        ), "Unexpected label_vector output size (2D)"

    @pytest.mark.parametrize("N", [1, 3, 6, 10, 100])
    def test_polynomial_vectorizer_get_label_vector(
        self, label_names, order, terms, terms_indexed, N
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        with pytest.raises(ValueError):
            _ = vec.get_label_vector(np.ones((N, len(vec.label_names) - 1)))
        with pytest.raises(ValueError):
            _ = vec.get_label_vector(np.ones((N, len(vec.label_names) + 1)))
        with pytest.raises(ValueError):
            vec.get_label_vector(np.ones((N, N)))

    def test_polynomial_vectorizer_get_label_vector_derivative(
        self, label_names, order, terms, terms_indexed
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        dt = vec.get_label_vector_derivative(
            np.asarray(range(len(vec.label_names))) + 1
        )
        assert dt.shape == (
            len(vec.terms) + 1,
            len(vec.label_names),
        ), "Wrong return shape"

    @pytest.mark.parametrize(
        "bad_input",
        (
            [],  # Too short
            np.arange(100),  # Too long
            [[1, 2]],  # Bad dimensions
        ),
    )
    def test_polynomial_vectorizer_get_label_vector_derivative_bad_input(
        self, label_names, order, terms, terms_indexed, bad_input
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        with pytest.raises(ValueError):
            _ = vec.get_label_vector_derivative(bad_input)

    @pytest.mark.parametrize("mul", ["*", "x"])
    @pytest.mark.parametrize("pow", ["^", "**"])
    @pytest.mark.parametrize("bracket", [True, False])
    def test_polynomial_vectorizer_get_human_readable_label_vector(
        self, label_names, order, terms, terms_indexed, mul, pow, bracket
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)

        assert vec.get_human_readable_label_vector(
            mul=mul, pow=pow, bracket=bracket
        ) == human_readable_label_vector(
            vec.terms, vec.label_names, mul=mul, pow=pow, bracket=bracket
        ), "get_human_readable_label_vector not mapped to instance method correctly"

    def test_polynomial_vectorizer_human_readable_label_vector(
        self, label_names, order, terms, terms_indexed
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)
        assert (
            vec.human_readable_label_vector == vec.get_human_readable_label_vector()
        ), "human_readable_label_vector property not correctly mapped"

    def test_polynomial_vectorizer_get_human_readable_label_term(
        self, label_names, order, terms, terms_indexed
    ):
        vec = PolynomialVectorizer(label_names=label_names, order=order)

        assert (
            vec.get_human_readable_label_term(0) == "1"
        ), "Did not return correct term for term_index = 0"

        for i, t in enumerate(vec.terms):
            assert vec.get_human_readable_label_term(
                i + 1
            ) == human_readable_label_term(
                t, label_names=vec.label_names
            ), "human_readable_label_term not mapped correctly for label_terms=None"
            assert vec.get_human_readable_label_term(
                i + 1, label_names=["a"] * len(vec.label_names)
            ) == human_readable_label_term(
                t, label_names=["a"] * len(vec.label_names)
            ), "human_readable_label_term not mapped correctly for label_terms specified"


@pytest.mark.parametrize(
    "input,output",
    [
        ("a", False),
        (1, False),
        (1.0, False),
        (1e3, False),
        ([], False),
        (
            [
                [],
                [],
            ],
            False,
        ),
        (["a", ("a", 3)], False),
        ([("a", 3), "a"], False),
        (
            [
                [
                    ("a", 4),
                ],
                ("b", 6),
            ],
            False,
        ),
        (
            [
                [
                    ("a", 4),
                ],
                [
                    ("b", 6),
                ],
            ],
            True,
        ),
        (
            [
                [
                    ("a", 4),
                    ("c", 1),
                ],
                [
                    ("b", 6),
                ],
            ],
            True,
        ),
    ],
)
def test__is_structured_label_vector(input, output):
    assert (
        _is_structured_label_vector(input) == output
    ), "_is_structured_label_vector output wrong"


@pytest.mark.parametrize(
    "description,label_vector,kwargs",
    [
        (
            [
                [
                    ("a", 4),
                    ("c", 1),
                ],
                [
                    ("b", 6),
                ],
            ],
            [
                [
                    ("a", 4),
                    ("c", 1),
                ],
                [
                    ("b", 6),
                ],
            ],
            {},
        ),  # Pass-through
        (
            "a^2 + a * b + b^2",
            [[("a", 2)], [("a", 1), ("b", 1)], [("b", 2)]],
            {},
        ),  # Basic
        (
            "a^2 + a * b + b^2 + b^0",
            [[("a", 2)], [("a", 1), ("b", 1)], [("b", 2)]],
            {},
        ),  # Drop 0-powers
        (
            "a^2+a*b+b^2",
            [[("a", 2)], [("a", 1), ("b", 1)], [("b", 2)]],
            {},
        ),  # No whitespace
        (
            "a**2 p axb p b**2",
            [[("a", 2)], [("a", 1), ("b", 1)], [("b", 2)]],
            {"sep": "p", "mul": "x", "pow": "**"},
        ),  # Secret kwargs
    ],
)
def test_parse_label_vector_description(description, label_vector, kwargs):
    assert (
        parse_label_vector_description(description, **kwargs) == label_vector
    ), "Unexpected return from parse_label_vector_description"


@pytest.mark.parametrize(
    "description",
    [
        "a^2 + a * b + b^n",  # Rubbish power
        "",  # No valid terms provided
    ],
)
def test_parse_label_vector_description_bad(description):
    with pytest.raises(ValueError):
        _ = parse_label_vector_description(description)


@mock.patch("numpy.isfinite", return_value=False)
def test_parse_label_vector_description_infinite_powers(isfinite):
    with pytest.raises(ValueError, match="non-finite"):
        _ = parse_label_vector_description("a^2")


@pytest.mark.parametrize(
    "term",
    [
        ("a", 4),  # Not deep enough
        "a^4",  # Not a structure term
        [("a", 4), "b", 6],  # Broken term
        [
            ("a", 1, 4),
        ],  # Too many parts
    ],
)
def test_human_readable_label_term_bad_input(term):
    with pytest.raises(ValueError, match="valid structured term"):
        _ = human_readable_label_term(term)


@pytest.mark.parametrize(
    "term",
    [
        [(1, 4)],
        [(1, 3), (0, 9)],
    ],
)
def test_human_readable_label_term_no_names(term):
    with pytest.raises(ValueError, match="Need label_names"):
        _ = human_readable_label_term(term)


@pytest.mark.parametrize("term,label_names,mul,pow,bracket,ret", [
    ([("a", 2)], None, "*", "^", False, "a^2"),
    ([("a", 2), ("b", 4)], None, "*", "^", False, "a^2*b^4"),
    ([("a", 2), ("b", 4)], None, "x", "**", False, "a**2xb**4"),
    ([("a", 2), ("b", 4)], None, "*", "^", True, "(a^2*b^4)"),
    ([("a", 2), ("b", 4)], None, "x", "**", True, "(a**2xb**4)"),
    ([("a", 2)], ["a", "b"], "*", "^", False, "a^2"),
    ([("a", 2), ("b", 4)], ["a", "b"], "*", "^", False, "a^2*b^4"),
    ([("a", 2), ("b", 4)], ["a", "b"], "x", "**", False, "a**2xb**4"),
    ([("a", 2), ("b", 4)], ["a", "b"], "*", "^", True, "(a^2*b^4)"),
    ([("a", 2), ("b", 4)], ["a", "b"], "x", "**", True, "(a**2xb**4)"),
    ([(0, 2)], ["a", "b"], "*", "^", False, "a^2"),
    ([(0, 2), (1, 4)], ["a", "b"], "*", "^", False, "a^2*b^4"),
    ([(0, 2), (1, 4)], ["a", "b"], "x", "**", False, "a**2xb**4"),
    ([(0, 2), (1, 4)], ["a", "b"], "*", "^", True, "(a^2*b^4)"),
    ([(0, 2), (1, 4)], ["a", "b"], "x", "**", True, "(a**2xb**4)"),
])
def test_human_readable_label_term(term, label_names, mul, pow, bracket, ret):
    assert (
        human_readable_label_term(
            term, label_names=label_names, mul=mul, pow=pow, bracket=bracket
        )
        == ret
    ), "human_readable_label_term gave wrong return"
