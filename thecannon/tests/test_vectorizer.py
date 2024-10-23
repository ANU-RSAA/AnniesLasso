"""
Unit tests for `thecannon.vectorizer`.
"""

import pytest

from ..vectorizer.base import BaseVectorizer
from ..vectorizer.polynomial import PolynomialVectorizer, terminator


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
