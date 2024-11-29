"""
Unit tests for `thecannon.vectorizer`.
"""

import pytest

from ..vectorizer.base import BaseVectorizer
from ..vectorizer.polynomial import PolynomialVectorizer, terminator


@pytest.mark.parametrize("vectorizer", [
    BaseVectorizer, PolynomialVectorizer
])
class TestVectorizersCommon:
    @pytest.mark.parametrize("label_names,terms,terms_out", [
        [("a", "b", "c"), [[("a", 0)], [("b", 1)], [("c", 2)]], None],
        [["a", "b", "c"], [[(0, 0)], [(1, 1)], [(2, 2)]], None],
        (["Teff", "g"], [[(0, 1)], [(0,2), (1, 1)], [(1, 2), (0, 2)]], None),
    ])
    def test_vectorizer_basic_init(self, vectorizer, label_names, terms, terms_out):
        vec = vectorizer(label_names=label_names, terms=terms)
        assert vec.label_names == tuple(label_names), "Label names not initialized correctly"
        assert vec.terms == (terms_out if terms_out is not None else terms), "Terms not initialized correctly"

    @pytest.mark.parametrize("label_names,terms", [
        [("a", "b"), [[("b", 1)]]],  # Unused term
        [("a", ), [[("a", 1), ("b", 1), ("c", 1)]]],  # Excess term (term number mismatch)
        [("a", "b"), [[("c", 1), ("a", 1)]]],  # Invalid term (str)
        [("a", "b"), [[(2, 1), (0, 1)]]],  # Invalid term (int)
    ])
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
        assert "label_names" in state[1].keys(), "Could not ID label_names in state return!"
        assert "terms" in state[1].keys(), "Could not ID terms in state return!"
        assert "metadata" in state[1].keys(), "Could not ID metadata in state return!"

    def test_vectorizer__setstate__(self, vectorizer):
        vec = vectorizer(label_names=("a"), terms=[[("a", 1)]])
        blank_vec = vectorizer(label_names=(), terms=[])
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


@pytest.mark.parametrize("label_names,terms,terms_out,order", [
    [("a", "b", "c"), "a^3 + b + c^2", [[(0, 3)], [(1, 1)], [(2, 2)]], None],
    (["Teff", "g"], "Teff + Teff^2*g + g^2*Teff^2", [[(0, 1)], [(0,2), (1, 1)], [(1, 2), (0, 2)]], None),
])
def test_polynomial_vectorizer_basic_init(label_names, terms, terms_out, order):
    vec = PolynomialVectorizer(label_names=label_names, order=order, terms=terms)
    assert vec.label_names == tuple(label_names), "Label names not initialized correctly"
    assert vec.terms == (terms_out if terms_out is not None else terms), "Terms not initialized correctly"

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


@pytest.mark.parametrize("sep,mul,pow", [
    ("s", "x", "p"),
])
def test_terminator_format_kwargs(sep, mul, pow):
    ret = terminator(["a", "b"], 2, -1, sep=sep, mul=mul, pow=pow)
    default_ret = terminator(["a", "b"], 2, -1)
    assert ret == default_ret.replace("+", sep).replace("*", mul).replace("^", pow)

@pytest.mark.parametrize("bad_cross_term_order", [
    "a", 19.2, "11",
    # -44.8  # This is actually OK - anything that succeeds the < 0 comparison will be set to order - 1
])
def test_terminator_bad_cross_term_order(bad_cross_term_order):
    with pytest.raises(TypeError):
        _ = terminator(["a", "b"], 2, bad_cross_term_order)
