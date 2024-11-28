"""
Unit tests for `thecannon.vectorizer`.
"""

import pytest

from ..vectorizer.base import BaseVectorizer
from ..vectorizer.polynomial import PolynomialVectorizer, terminator


@pytest.mark.parametrize("vectorizer", [
    BaseVectorizer, PolynomialVectorizer
])
@pytest.mark.parametrize("label_names,terms,terms_out", [
    [("a", "b", "c"), [[("a", 0)], [("b", 1)], [("c", 2)]], None],
    [["a", "b", "c"], [[(0, 0)], [(1, 1)], [(2, 2)]], None],
    (["Teff", "g"], [[(0, 1)], [(0,2), (1, 1)], [(1, 2), (0, 2)]], None),
])
def test_vectorizer_basic_init(vectorizer, label_names, terms, terms_out):
    vec = vectorizer(label_names=label_names, terms=terms)
    assert vec.label_names == tuple(label_names), "Label names not initialized correctly"
    assert vec.terms == (terms_out if terms_out is not None else terms), "Terms not initialized correctly"

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
