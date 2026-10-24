# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the check that no value of .env is in a tracked file
(dev/check_env_secrets.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dev import check_env_secrets
from dev.check_env_secrets import find_secrets, main, read_secrets


# A value of the test .env files: long enough to be searched for
_KEY = "sk-test-0123456789abcdef"


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Returns a .env file in a temporary folder, which the check reads.
    """
    path = tmp_path / ".env"
    path.write_text(
        f"MY_API_KEY={_KEY}\n"
        'MY_API_BASE="https://my-resource.example.com/"\n'
        "SHORT=DUMMY\n"
        "EMPTY=\n"
        "# COMMENTED=sk-commented-0123456789\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_env_secrets, "ENV_FILE", path)
    return path


def test_read_secrets(env_file: Path, tmp_path: Path) -> None:
    """
    The values of .env are searched for as they are written, and a URL also
    without its trailing slash. Short and empty values, and commented lines,
    are left out. Without a .env file, nothing is searched for.
    """
    assert read_secrets() == {
        "MY_API_KEY": [_KEY.encode()],
        "MY_API_BASE": [
            b"https://my-resource.example.com/",
            b"https://my-resource.example.com",
        ],
    }
    assert read_secrets(tmp_path / "missing.env") == {}


def test_find_secrets(env_file: Path, tmp_path: Path) -> None:
    """
    A file that has a value is found, with the line of its first occurrence and
    the name of its variable, binary files included, and a URL also where it is
    written without its trailing slash, e.g. as the start of a longer URL.
    """
    with_key = tmp_path / "with_key.py"
    with_key.write_text(f"a = 1\nkey = '{_KEY}'\nb = '{_KEY}'\n", encoding="utf-8")
    with_url = tmp_path / "cassette.yaml"
    with_url.write_text(
        "uri: https://my-resource.example.com/providers/x\n", encoding="utf-8"
    )
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"\x00\xff" + _KEY.encode() + b"\x00")
    clean = tmp_path / "clean.txt"
    clean.write_text("DUMMY https://other.example.com\n", encoding="utf-8")
    assert find_secrets(
        [with_key, with_url, binary, clean, tmp_path / "missing.txt"], read_secrets()
    ) == [
        (with_key, 2, "MY_API_KEY"),
        (with_url, 1, "MY_API_BASE"),
        (binary, 1, "MY_API_KEY"),
    ]


def test_main(
    env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    The check fails for a file that has a value, and reports the file, the line
    and the name of the variable, never the value. It passes for files without
    a value, and checks nothing without a .env file.
    """
    leaked = tmp_path / "leaked.txt"
    leaked.write_text(f"token: {_KEY}\n", encoding="utf-8")
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing here\n", encoding="utf-8")

    assert main([str(clean)]) == 0
    assert main([str(leaked), str(clean)]) == 1
    output = capsys.readouterr().out
    assert f"{leaked}:1: the value of MY_API_KEY from .env" in output
    assert str(clean) not in output
    assert _KEY not in output

    env_file.unlink()
    assert main([str(leaked)]) == 0
