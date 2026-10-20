# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Runs the tests of ``Document.from_docling()`` with the oldest docling-core that
``pyproject.toml`` allows.

The tests run with the versions of ``uv.lock``. The lower bound of docling-core in
the ``docling`` extra, which the converter checks (``_MIN_DOCLING_CORE_VERSION`` in
``contextgem/internal/converters/docling.py``), says that ``Document.from_docling()``
works with that version too, e.g. for a user whose other packages hold docling-core
back. This script checks it: it creates a throwaway environment from ``uv.lock``,
with the dev group, replaces docling-core with its lower bound, and runs the tests of
``Document.from_docling()`` there.

The environment has docling-core only, without Docling, as for a Docling document
converted elsewhere: the docling-slim of the dev group needs a newer docling-core
than the lower bound, so it is uninstalled. The test that converts the fixture files
with it, ``test_from_docling_converted_files``, is left out; the others read the
committed Docling documents of the fixtures. ``uv pip check`` checks that what is
installed is consistent before the tests run.

Takes a minute or two, and sends no LLM request. The workflow
``.github/workflows/ci-tests.yml`` runs it on each push.

Example:
    uv run fab docling-floor
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404 - runs uv and Python with arguments of its own, no shell
import sys
import tempfile
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The tests of Document.from_docling(), without the one that converts files with
# Docling
TEST_FILES = ["tests/test_documents.py", "tests/test_docling_properties.py"]
TEST_SELECTION = "docling and not test_from_docling_converted_files"


def _docling_core_lower_bound() -> str:
    """
    Read the lower bound of docling-core in the ``docling`` extra of
    ``pyproject.toml``.

    :returns: The requirement of docling-core at its lower bound, e.g.
        ``docling-core==2.103.0``.
    :rtype: str
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for requirement in project["project"]["optional-dependencies"]["docling"]:
        if requirement.startswith("docling-core "):
            lower_bound = re.search(r">=\s*([\d.]+)", requirement)
            if lower_bound is None:
                sys.exit("docling-core has no lower bound (>=) in the docling extra.")
            return f"docling-core=={lower_bound.group(1)}"
    sys.exit("docling-core is not in the docling extra of pyproject.toml.")


def _setup(scratch: Path, docling_core: str) -> Path:
    """
    Create a throwaway environment from ``uv.lock``, with the dev group, with
    a version of docling-core and without Docling.

    :param scratch: A folder to create the environment in, outside the checkout.
    :type scratch: Path
    :param docling_core: The requirement of docling-core, e.g.
        ``docling-core==2.103.0``.
    :type docling_core: str
    :returns: The Python of the environment.
    :rtype: Path
    """
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv was not found.")
    venv = scratch / "venv"
    # Without the environment that `uv run` started this script in
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    print("Creating an environment from uv.lock...", flush=True)
    subprocess.run(  # nosec B603 - uv with arguments of its own, no shell
        [uv, "sync", "--locked", "--all-groups", "--quiet"],
        cwd=ROOT,
        env={**env, "UV_PROJECT_ENVIRONMENT": str(venv)},
        check=True,
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    target = ("--python", str(python))
    print(f"Installing {docling_core}, without Docling...", flush=True)
    for command in (
        ("uninstall", "--quiet", *target, "docling-slim"),
        ("install", "--quiet", *target, docling_core),
        # Fails if a package needs another docling-core, i.e. if the tests would run
        # in an environment that no install gives
        ("check", "--quiet", *target),
    ):
        subprocess.run([uv, "pip", *command], env=env, check=True)  # nosec B603 - uv with arguments of its own, no shell
    return python


def main() -> int:
    """
    Run the tests of ``Document.from_docling()`` with the lower bound of
    docling-core.

    :returns: The exit code of pytest.
    :rtype: int
    """
    docling_core = _docling_core_lower_bound()
    # Outside the checkout. A file of the environment can still be open on Windows
    # when the folder is removed.
    with tempfile.TemporaryDirectory(
        prefix="contextgem-docling-floor-", ignore_cleanup_errors=True
    ) as scratch:
        python = _setup(Path(scratch), docling_core)
        return subprocess.run(  # nosec B603 - the Python of the environment, no shell
            [
                str(python),
                *("-m", "pytest", *TEST_FILES),
                *("-k", TEST_SELECTION),
                *("-p", "no:cacheprovider"),
                "-q",
            ],
            cwd=ROOT,
            check=False,
        ).returncode


if __name__ == "__main__":
    sys.exit(main())
