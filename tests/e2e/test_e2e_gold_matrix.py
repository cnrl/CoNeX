"""dtype/seed matrix over the golden scenario families.

Every scenario family runs over dtype float32/float64 and seeds 0, 1, 2
and is checked for: determinism (same seed twice => equal), seed
sensitivity (different seed => different, for the families that draw
random numbers), and finiteness (no NaN/inf in any reported value after
the scenario's steps).
"""

import math

import pytest
import torch

from .golden._scenarios import SCENARIOS

SEEDS = (0, 1, 2)
DTYPES = ("float32", "float64")
# families whose setup draws random numbers (SP potential pools, grid
# scales/orientations); the rest are fully deterministic in this scenario.
# temporal_memory is deterministic here: its randperm only fires with
# multiple candidate synapses, which this tiny sequence never produces.
RANDOM_FAMILIES = ("spatial_pooler", "grid_cell")


def leaves(value):
    """Yield every number nested in a scenario result."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from leaves(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)


def equal(a, b):
    """Deep equality: exact for int/bool, allclose for floats."""
    if isinstance(a, bool) or isinstance(a, int):
        return a == b
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict):
        return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    raise AssertionError(f"unsupported type {type(a)}")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_deterministic(name, seed, dtype_name):
    """Running the scenario twice with the same seed gives equal results."""
    dtype = getattr(torch, dtype_name)
    first = SCENARIOS[name](seed, dtype)
    second = SCENARIOS[name](seed, dtype)
    assert equal(first, second), f"{name} seed={seed} {dtype_name} not equal"


@pytest.mark.parametrize("name", sorted(RANDOM_FAMILIES))
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_seed_sensitive(name, dtype_name):
    """A different seed changes the result where randomness exists."""
    dtype = getattr(torch, dtype_name)
    a = SCENARIOS[name](0, dtype)
    b = SCENARIOS[name](1, dtype)
    assert not equal(a, b), f"{name} {dtype_name}: seeds 0 and 1 agree"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_finite_after_steps(name, seed, dtype_name):
    """No NaN/inf in any value the scenario reports after its steps."""
    dtype = getattr(torch, dtype_name)
    result = SCENARIOS[name](seed, dtype)
    bad = [v for v in leaves(result) if not math.isfinite(v)]
    assert not bad, f"{name} seed={seed} {dtype_name}: non-finite {bad}"
