#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from thecannon import model
from unittest import mock


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
