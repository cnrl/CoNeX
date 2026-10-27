#!/usr/bin/env python

"""Tests for `CoNeX` package."""

import pytest
import conex


def test_import():
    """Test that the package can be imported."""
    assert hasattr(conex, '__version__')
    assert conex.__version__ == "0.1.5"
