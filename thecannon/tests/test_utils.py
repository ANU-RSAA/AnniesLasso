#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for general utility functions.
"""

import pytest

from thecannon import utils


def test_short_hash_different_boolean():
    assert utils.short_hash(True) != utils.short_hash(
        False
    ), "hash match for True and False"


@pytest.mark.parametrize(
    "input,hash",
    [
        ("ABCDE", "7fc56270e79d5ed678fe0d61f8370cf623e75af33a3ea00cfc"),
        (
            "1234567890",
            "c4ca4238a0c81e728d9deccbc87e4ba87ff679a2e4da3b7fbb1679091c5a8f14e45fcec9f0f895fb45c48cce2ecfcd208495",
        ),
        ("a1b2c3", "0cc175b9c0c4ca4238a092eb5ffee6c81e728d9d4a8a08f09deccbc87e4b"),
        ("az09AZ", "0cc175b9c0fbade9e36acfcd20849545c48cce2e7fc56270e721c2e59531"),
    ],
)
def test_hash_consistency(input, hash):
    """Ensure hashing does not change with time and break old hashes."""
    assert utils.short_hash(input) == hash, "Hash value has changed!"
