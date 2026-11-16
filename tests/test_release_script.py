"""Tests for scripts/check_release.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_release import check, main, parse_version_from_setup  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SETUP_PY = """\
from setuptools import setup

setup(
    name="fake-pkg",
    version="1.2.3",
)
"""

HISTORY_WITH_ENTRY = """\
=======
History
=======

1.2.3 (2026-01-01)
------------------

- A change.
"""

HISTORY_WITHOUT_ENTRY = """\
=======
History
=======

1.2.2 (2025-12-01)
------------------

- An older change.
"""


def make_repo(tmp_path, setup_text=SETUP_PY, history_text=HISTORY_WITH_ENTRY):
    (tmp_path / "setup.py").write_text(setup_text, encoding="utf-8")
    (tmp_path / "HISTORY.rst").write_text(history_text, encoding="utf-8")
    return tmp_path


def test_match_with_v(tmp_path):
    make_repo(tmp_path)
    ok, message = check("v1.2.3", tmp_path / "setup.py", tmp_path / "HISTORY.rst")
    assert ok is True
    assert "matches" in message


def test_match_without_v(tmp_path):
    make_repo(tmp_path)
    ok, _ = check("1.2.3", tmp_path / "setup.py", tmp_path / "HISTORY.rst")
    assert ok is True


def test_mismatch(tmp_path):
    make_repo(tmp_path)
    ok, message = check("v1.2.4", tmp_path / "setup.py", tmp_path / "HISTORY.rst")
    assert ok is False
    assert "does not match" in message


@pytest.mark.parametrize("tag", ["not-a-version", "v1.2", "1.2.3.4", "v", ""])
def test_bad_tag(tmp_path, tag):
    make_repo(tmp_path)
    code = main([tag])
    assert code == 2


def test_missing_history_entry(tmp_path):
    make_repo(tmp_path, history_text=HISTORY_WITHOUT_ENTRY)
    ok, message = check("v1.2.3", tmp_path / "setup.py", tmp_path / "HISTORY.rst")
    assert ok is False
    assert "no entry" in message


def test_real_repo_current_version():
    """Real repo files: tag matches setup.py, but HISTORY.rst has no entries."""
    version = parse_version_from_setup(REPO_ROOT / "setup.py")
    assert version == "0.1.5"
    ok, message = check("v0.1.5", REPO_ROOT / "setup.py", REPO_ROOT / "HISTORY.rst")
    # HISTORY.rst currently has no version entries at all, so the history
    # check fails even though the tag matches setup.py.
    assert ok is False
    assert "no entry" in message
