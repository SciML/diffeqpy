"""Tests for the warmup module."""

import pytest


def test_warmup_ode():
    """Test that warmup_ode runs without error."""
    from diffeqpy.warmup import warmup_ode

    elapsed = warmup_ode(verbose=False)
    assert isinstance(elapsed, float)
    assert elapsed > 0


def test_warmup_de():
    """Test that warmup_de runs without error."""
    from diffeqpy.warmup import warmup_de

    elapsed = warmup_de(verbose=False)
    assert isinstance(elapsed, float)
    assert elapsed > 0
