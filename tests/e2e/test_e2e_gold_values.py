"""Golden-number tests: re-run each scenario and compare with values.json.

Tolerances: float32 scenarios use rtol=1e-5, atol=1e-6 (float32 eps is
~1.2e-7, so this is ~100 eps, far below any behavior-level change);
float64 uses rtol=1e-12, atol=1e-14. Integers, booleans and index lists
are compared exactly. The json is keyed by torch major version; the test
compares against the section of the running torch.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from .golden._scenarios import SCENARIOS

HERE = Path(__file__).resolve().parent
VALUES = HERE / "golden" / "values.json"
GENERATE = HERE / "golden" / "generate.py"
REPO_ROOT = HERE.parent.parent
SEEDS = (0, 1, 2)
# (dtype, rtol, atol) per scenario dtype
DTYPES = {
    "float32": (torch.float32, 1e-5, 1e-6),
    "float64": (torch.float64, 1e-12, 1e-14),
}
MAJOR = torch.__version__.split(".")[0]


def section():
    with open(VALUES) as f:
        doc = json.load(f)
    assert MAJOR in doc, f"values.json has no section for torch {MAJOR}"
    return doc[MAJOR]


def compare(name, fresh, expected, rtol, atol):
    """Compare one scenario result against its committed golden value."""
    if isinstance(expected, bool) or isinstance(expected, int):
        assert fresh == expected, f"{name}: {fresh} != {expected}"
        return
    if isinstance(expected, list) and expected and all(
        isinstance(v, (bool, int)) for v in expected
    ):
        assert fresh == expected, f"{name}: {fresh} != {expected}"
        return
    if isinstance(expected, list) and expected and all(
        isinstance(v, float) for v in expected
    ):
        got = torch.tensor(fresh)
        want = torch.tensor(expected)
        assert torch.allclose(got, want, rtol=rtol, atol=atol), (
            f"{name}: max diff {(got - want).abs().max().item()}")
        return
    if isinstance(expected, float):
        assert torch.allclose(torch.tensor(fresh), torch.tensor(expected),
                              rtol=rtol, atol=atol), (
            f"{name}: {fresh} vs {expected}")
        return
    if isinstance(expected, dict):
        assert set(fresh) == set(expected), f"{name}: keys {fresh} vs {expected}"
        for k in expected:
            compare(f"{name}.{k}", fresh[k], expected[k], rtol, atol)
        return
    if isinstance(expected, list):
        for i, (g, w) in enumerate(zip(fresh, expected)):
            compare(f"{name}[{i}]", g, w, rtol, atol)
        return
    raise AssertionError(f"{name}: unsupported golden type {type(expected)}")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype_name", sorted(DTYPES))
def test_golden_value(name, seed, dtype_name):
    """Re-running the scenario reproduces the committed golden number."""
    dtype, rtol, atol = DTYPES[dtype_name]
    key = f"{name}|seed={seed}|{dtype_name}"
    fresh = SCENARIOS[name](seed, dtype)
    compare(key, fresh, section()[key], rtol, atol)


def test_generator_output_matches_committed_json():
    """generate.py without --write must exit 0: the json is not hand-edited."""
    proc = subprocess.run(
        [sys.executable, str(GENERATE)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "matches the running library" in proc.stdout


def test_generator_rewrite_matches_committed(tmp_path):
    """A --write regeneration matches the committed json (within tolerance).

    We compare the parsed running-torch section with tolerance, not raw bytes:
    float formatting can differ across platforms (different BLAS/libm), so a
    byte comparison would be brittle.
    """
    from .golden.generate import sections_close

    target = tmp_path / "values.json"
    target.write_bytes(VALUES.read_bytes())
    proc = subprocess.run(
        [sys.executable, str(GENERATE), "--write", "--out", str(target)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(target) as f:
        regenerated = json.load(f)
    committed = section()
    assert sections_close(committed, regenerated[MAJOR]), (
        "regenerated section differs from committed beyond tolerance")
