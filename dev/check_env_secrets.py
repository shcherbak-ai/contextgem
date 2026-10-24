# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Checks that no value of the ``.env`` file, e.g. an API key of the tests, is in
a file that git tracks.

``.env``, which git ignores, holds the API keys and the endpoints with which
the tests record their cassettes, and the recorded requests are redacted before
they are written. This check catches a value that a redaction misses, or that
reaches a tracked file otherwise, e.g. a test or a log pasted into a file.

Each value of ``.env`` is searched for as it is written, in the bytes of each
file, binary files included, and so is a URL without its trailing slash. A value
shorter than ``MIN_SECRET_LENGTH`` characters is left out, e.g. ``DUMMY``, which
the tests set on purpose. The check reports the variable and the line of the file
that has its value, never the value. Without a ``.env`` file, e.g. in CI, there
is nothing to check.

pre-commit gives it the staged files at a commit, and all the tracked files with
``--all-files``. Without arguments, it checks all the files that git tracks.
"""

from __future__ import annotations

import subprocess  # nosec B404 - runs git with arguments of its own, no shell
import sys
from collections.abc import Iterable
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

# The length below which a value is not searched for: such a value, e.g. ``true``
# or ``DUMMY``, is no secret, and would be found in many files
MIN_SECRET_LENGTH = 8


def read_secrets(env_file: Path | None = None) -> dict[str, list[bytes]]:
    """
    Read the values of the ``.env`` file to search for.

    :param env_file: The ``.env`` file, by default ``ENV_FILE``.
    :type env_file: Path | None
    :returns: The forms of each value to search for, by the name of its
        variable: the value, and for a URL with a trailing slash, the URL without
        it. Empty if the file does not exist.
    :rtype: dict[str, list[bytes]]
    """
    env_file = ENV_FILE if env_file is None else env_file
    if not env_file.is_file():
        return {}
    secrets: dict[str, list[bytes]] = {}
    for name, value in dotenv_values(env_file).items():
        value = (value or "").strip()
        if len(value) < MIN_SECRET_LENGTH:
            continue
        forms = [value]
        if "://" in value and value.endswith("/"):
            forms.append(value.rstrip("/"))
        secrets[name] = [form.encode("utf-8") for form in forms]
    return secrets


def find_secrets(
    paths: Iterable[Path], secrets: dict[str, list[bytes]]
) -> list[tuple[Path, int, str]]:
    """
    Find the values in the files.

    :param paths: The files to search.
    :type paths: Iterable[Path]
    :param secrets: The forms of each value, by the name of its variable (see
        ``read_secrets``).
    :type secrets: dict[str, list[bytes]]
    :returns: Each file that has a value, with the line of its first occurrence
        and the name of the variable, in the order of the files.
    :rtype: list[tuple[Path, int, str]]
    """
    findings: list[tuple[Path, int, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        for name, forms in secrets.items():
            positions = [data.find(form) for form in forms if form in data]
            if positions:
                line = data.count(b"\n", 0, min(positions)) + 1
                findings.append((path, line, name))
    return findings


def tracked_files() -> list[Path]:
    """
    List the files that git tracks.

    :returns: The files, with their paths in the working tree.
    :rtype: list[Path]
    :raises RuntimeError: If git fails.
    """
    result = subprocess.run(  # nosec B603 B607 - git, with arguments of its own
        ["git", "ls-files", "-z"],
        capture_output=True,
        cwd=ROOT,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {result.stderr.decode().strip()}")
    return [ROOT / name for name in result.stdout.decode("utf-8").split("\0") if name]


def main(arguments: Iterable[str] | None = None) -> int:
    """
    Check the files given, or all the tracked files without arguments.

    :param arguments: The paths of the files, by default those of the command
        line.
    :type arguments: Iterable[str] | None
    :returns: The exit code: 1 if a file has a value of ``.env``, 0 otherwise.
    :rtype: int
    """
    names = list(sys.argv[1:] if arguments is None else arguments)
    secrets = read_secrets()
    if not secrets:
        return 0
    paths = [Path(name) for name in names] if names else tracked_files()
    findings = find_secrets(paths, secrets)
    for path, line, name in findings:
        print(f"{path}:{line}: the value of {name} from .env")
    if findings:
        print(
            "Remove these values from the files: a file that git tracks must not "
            "hold a value of .env. In a cassette, redact the value where it is "
            "recorded (tests/utils.py)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
