"""Regenerate tests/e2e/golden/values.json from the running library.

Usage:  python tests/e2e/golden/generate.py [--write]

The json is keyed by torch major version ({"1": {...}, "2": {...}}) so a
scenario that is nondeterministic across torch 1.13.1 and 2.x keeps both
entries. --write updates only the section for the running torch. Without
--write the script prints a diff of the running-torch section against the
committed json and exits 1 if they differ.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.e2e.golden._scenarios import SCENARIOS  # noqa: E402

HERE = Path(__file__).resolve().parent
VALUES = HERE / "values.json"
SEEDS = (0, 1, 2)
DTYPES = ("float32", "float64")


def collect():
    """Run every scenario over seeds x dtypes; keys are name|seed=dtype."""
    values = {}
    for name, fn in SCENARIOS.items():
        for seed in SEEDS:
            for dtype_name in DTYPES:
                dtype = getattr(torch, dtype_name)
                values[f"{name}|seed={seed}|{dtype_name}"] = fn(seed, dtype)
    return values


def load_committed(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="update the running torch's section in values.json")
    parser.add_argument("--out", type=Path, default=VALUES,
                        help="json file to read/write (default: values.json)")
    args = parser.parse_args()

    major = torch.__version__.split(".")[0]
    fresh = collect()
    if args.write:
        doc = load_committed(args.out) or {}
        doc[major] = fresh
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"updated {args.out} section {major!r} (torch {torch.__version__})")
        return 0

    committed = load_committed(args.out)
    if committed is None:
        print(f"no committed {args.out}; run with --write")
        return 1
    old = committed.get(major)
    if old is None:
        print(f"no section for torch {major} in {args.out}; run with --write")
        return 1
    if old == fresh:
        print(f"{args.out.name} section {major!r} matches the running library "
              f"(torch {torch.__version__})")
        return 0
    print(f"{args.out.name} section {major!r} differs from the running library "
          f"(torch {torch.__version__}):")
    for key in sorted(set(old) | set(fresh)):
        if old.get(key) != fresh.get(key):
            print(f"  {key}:\n    committed: {old.get(key)}\n    fresh:     "
                  f"{fresh.get(key)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
