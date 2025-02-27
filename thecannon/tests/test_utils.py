#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for general utility functions.
"""

import pytest

from thecannon import utils

def test_short_hash_different_boolean():
    assert utils.short_hash(True) != utils.short_hash(False), "hash match for True and False"

def test_hash_consistency():
    """Ensure hashing does not change with time and break old hashes."""
    assert utils.short_hash("ABCDE") == "7fc56270e79d5ed678fe0d61f8370cf623e75af33a3ea00cfc"
