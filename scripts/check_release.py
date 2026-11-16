#!/usr/bin/env python
"""Check that a release tag matches the version in setup.py and HISTORY.rst.

Usage: python scripts/check_release.py <tag>

Exit codes:
  0  tag matches the version in setup.py and HISTORY.rst has an entry
  1  tag differs from the setup.py version, or HISTORY.rst has no entry
  2  tag is not valid semantic-version style (or usage error)
"""

import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
SEMVER_RE = re.compile(r'^\d+\.\d+\.\d+$')


def parse_version_from_setup(setup_path):
    """Return the version string from the version= line in setup.py."""
    text = Path(setup_path).read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError(f"no version= line found in {setup_path}")
    return match.group(1)


def normalize_tag(tag):
    """Strip an optional leading 'v' from a tag. Raise ValueError if not
    semantic-version style (digits.digits.digits)."""
    candidate = tag[1:] if tag.startswith("v") else tag
    if not SEMVER_RE.match(candidate):
        raise ValueError(
            f"tag {tag!r} is not semantic-version style (expected vX.Y.Z or X.Y.Z)"
        )
    return candidate


def check(tag, setup_path, history_path):
    """Return (ok, message) for the tag against setup.py and HISTORY.rst."""
    version = parse_version_from_setup(setup_path)
    tag_version = normalize_tag(tag)
    if tag_version != version:
        return False, f"tag {tag!r} does not match setup.py version {version!r}"
    history = Path(history_path).read_text(encoding="utf-8")
    # whole-token match so e.g. 0.1.5 does not match inside 10.1.50
    if not re.search(rf"(?<![\d.]){re.escape(tag_version)}(?![\d.])", history):
        return False, f"HISTORY.rst has no entry for version {tag_version!r}"
    return True, f"tag {tag!r} matches version {version!r} and HISTORY.rst"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: check_release.py <tag>", file=sys.stderr)
        return 2
    tag = argv[0]
    repo_root = Path(__file__).resolve().parent.parent
    try:
        tag_version = normalize_tag(tag)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        ok, message = check(tag, repo_root / "setup.py", repo_root / "HISTORY.rst")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
