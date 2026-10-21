# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Checks that ContextGem installs on each platform and supported Python.

Resolves the dependencies of ``pyproject.toml``, or with ``--version`` a release
from PyPI, with all their extras, e.g. ``docling``, with ``uv pip compile
--only-binary :all: --python-platform <platform> --python-version <X.Y>``, i.e. as
an install without a compiler gets them, with the newest versions that the
requirements admit. uv answers for any platform from
the wheels that PyPI lists, so no machine of the platform is needed. Sends no LLM
request, and needs no API key.

Run it before a release, and after a change to the version or the range of a
dependency, while a gap can still be closed or documented before it ships; with
``--version``, after a release, for the package as PyPI serves it. ContextGem's
own wheel is pure Python: what installs on a platform is decided by its
dependencies.

The installs that are known to fail are listed in ``EXPECTED_FAILURES``, with
their reasons. The check reports any difference from the list, and then exits
with 1: a new failure, or an install that no longer fails, after which the list
and the documentation that names the gap are updated, e.g. the note on Alpine in
``docs/source/installation.rst``.

Example:
    uv run fab install-check
    uv run fab install-check --version latest
    uv run fab install-check --version 1.1.0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess  # nosec B404 - runs uv with arguments of its own, no shell
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The platforms, by uv's names for them
PLATFORMS = {
    "x86_64-manylinux_2_28": "Linux x86_64 (glibc)",
    "aarch64-manylinux_2_28": "Linux arm64 (glibc)",
    "x86_64-unknown-linux-musl": "Linux x86_64 (musl, Alpine)",
    "x86_64-pc-windows-msvc": "Windows x86_64",
    "aarch64-pc-windows-msvc": "Windows arm64",
    "aarch64-apple-darwin": "macOS arm64",
    "x86_64-apple-darwin": "macOS x86_64 (Intel)",
}
# The macOS version that the check assumes. Without it uv assumes an older one,
# for which onnxruntime publishes no wheels.
MACOS_TARGET = "14.0"

# The installs known to fail, by platform and Python (None: on every Python)
EXPECTED_FAILURES: dict[tuple[str, str | None], str] = {
    ("x86_64-unknown-linux-musl", None): (
        "onnxruntime publishes no wheels for musl (docs/source/installation.rst)"
    ),
    ("aarch64-pc-windows-msvc", None): (
        "litellm, tiktoken and fastuuid publish no wheels for Windows on arm64"
    ),
    ("x86_64-apple-darwin", "3.14"): (
        "onnxruntime 1.24+, needed on Python 3.14, publishes no Intel Mac wheels "
        "(docs/source/installation.rst)"
    ),
}

# The installs with the extras that are known to fail where ContextGem installs
# without them, by platform and Python (None: on every Python)
EXPECTED_EXTRA_FAILURES: dict[tuple[str, str | None], str] = {
    ("x86_64-apple-darwin", None): (
        "docling-parse 7+, needed by Docling (the docling extra), publishes "
        "no Intel Mac wheels (docs/source/installation.rst)"
    ),
}


def _expected(
    failures: dict[tuple[str, str | None], str], target: str, python: str
) -> str | None:
    """
    Look up why an install is known to fail.

    :param failures: The known failures, by platform and Python.
    :type failures: dict[tuple[str, str | None], str]
    :param target: The platform.
    :type target: str
    :param python: The Python version.
    :type python: str
    :returns: The reason, or None for an install that is expected to work.
    :rtype: str | None
    """
    return failures.get((target, python), failures.get((target, None)))


def _release(version: str | None) -> tuple[str, str, list[str]]:
    """
    Look the release up on PyPI.

    :param version: The version, or None for the latest release.
    :type version: str | None
    :returns: The version, its ``requires-python`` and its extras.
    :rtype: tuple[str, str, list[str]]
    """
    path = f"contextgem/{version}/json" if version else "contextgem/json"
    try:
        with urllib.request.urlopen(  # nosec B310 - a fixed https URL of PyPI
            f"https://pypi.org/pypi/{path}", timeout=30
        ) as response:
            info = json.load(response)["info"]
    except urllib.error.HTTPError as e:
        sys.exit(f"contextgem {version} is not on PyPI ({e.code}).")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach PyPI: {e.reason}")
    return (
        info["version"],
        info["requires_python"],
        list(info.get("provides_extra") or []),
    )


def _working_tree() -> tuple[str, str, list[str]]:
    """
    Read the version, ``requires-python`` and extras of ``pyproject.toml``.

    :returns: The version, its ``requires-python`` and its extras.
    :rtype: tuple[str, str, list[str]]
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    return (
        project["version"],
        project["requires-python"],
        list(project.get("optional-dependencies", {})),
    )


def _python_versions(requires_python: str) -> list[str]:
    """
    List the Python versions that a ``requires-python`` admits.

    :param requires_python: E.g. ``<3.15,>=3.11``.
    :type requires_python: str
    :returns: E.g. ``["3.11", "3.12", "3.13", "3.14"]``.
    :rtype: list[str]
    """
    low = re.search(r">=\s*3\.(\d+)", requires_python)
    high = re.search(r"<\s*3\.(\d+)", requires_python)
    if not low or not high:
        sys.exit(f"Cannot read the Python versions of {requires_python!r}.")
    return [f"3.{minor}" for minor in range(int(low[1]), int(high[1]))]


def _blocker(stderr: str) -> str:
    """
    Name what keeps uv from resolving the requirements.

    :param stderr: The error output of ``uv pip compile``.
    :type stderr: str
    :returns: The first package without a wheel, or the last line of the error.
    :rtype: str
    """
    match = re.search(
        r"([A-Za-z0-9_.-]+?)(?:[=<>!~]=?\S+)? has no (?:usable )?wheels", stderr
    )
    if match:
        return match[1]
    lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1][:100] if lines else "unknown error"


def _resolve(
    uv: str, sources: list[str], target: str, python: str, scratch: str
) -> str | None:
    """
    Resolve the requirements for a platform and a Python, from wheels only.

    :param uv: The path of the uv command.
    :type uv: str
    :param sources: The arguments that give ``uv pip compile`` the requirements.
    :type sources: list[str]
    :param target: The platform, by uv's name for it.
    :type target: str
    :param python: The Python version.
    :type python: str
    :param scratch: A folder to work in, outside the checkout.
    :type scratch: str
    :returns: None when they resolve, else what keeps them from resolving.
    :rtype: str | None
    """
    proc = subprocess.run(  # nosec B603 - uv with arguments of its own, no shell
        [
            *(uv, "pip", "compile", "--quiet", "--no-config", "--no-header"),
            *("--python-platform", target, "--python-version", python),
            *("--only-binary", ":all:", *sources),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=scratch,
        env={**os.environ, "MACOSX_DEPLOYMENT_TARGET": MACOS_TARGET},
        check=False,
    )
    return None if proc.returncode == 0 else _blocker(proc.stderr)


def main() -> int:
    """
    Resolve the requirements on each platform and Python, and print the table.

    :returns: 0 when every install went as expected, else 1.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Check that ContextGem installs on each platform and "
        "supported Python, from wheels only: the dependencies of pyproject.toml, "
        "or a release from PyPI. Sends no LLM request."
    )
    parser.add_argument(
        "--version", help="a release on PyPI, or 'latest' (default: pyproject.toml)"
    )
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv was not found.")

    # Outside the checkout, so that no configuration of the project applies
    with tempfile.TemporaryDirectory(prefix="contextgem-check-install-") as scratch:
        if args.version:
            version, requires_python, extras = _release(
                None if args.version == "latest" else args.version
            )
            subject = f"contextgem {version} from PyPI"

            def sources(with_extras: bool) -> list[str]:
                """
                Write the requirement of the release, with or without its extras.

                :param with_extras: Whether to require the extras.
                :type with_extras: bool
                :returns: The arguments that give ``uv pip compile`` the requirement.
                :rtype: list[str]
                """
                requirement = "contextgem"
                if with_extras:
                    requirement += f"[{','.join(extras)}]"
                path = Path(scratch) / f"requirements-{int(with_extras)}.in"
                path.write_text(f"{requirement}=={version}\n", encoding="utf-8")
                # PyPI lets its index be cached for 10 minutes: without the refresh,
                # a release published since the last check can be missing
                return ["--refresh-package", "contextgem", str(path)]

        else:
            version, requires_python, extras = _working_tree()
            subject = f"The dependencies of pyproject.toml (contextgem {version})"

            def sources(with_extras: bool) -> list[str]:
                """
                Give the dependencies of ``pyproject.toml``, with or without its extras.

                :param with_extras: Whether to include the extras.
                :type with_extras: bool
                :returns: The arguments that give ``uv pip compile`` the requirements.
                :rtype: list[str]
                """
                return [*(["--all-extras"] if with_extras else []), str(PYPROJECT)]

        pythons = _python_versions(requires_python)
        # Without the extras, then with them, where ContextGem has extras
        variants = [(False, subject)]
        if extras:
            plural = "s" if len(extras) > 1 else ""
            variants.append((True, f"With the extra{plural} {', '.join(extras)}"))
        cells = [
            (with_extras, target, python)
            for with_extras, _ in variants
            for target in PLATFORMS
            for python in pythons
        ]
        # Once for each variant, before the resolutions: a requirements file written
        # while another resolution reads it can be read empty, which resolves
        arguments = {with_extras: sources(with_extras) for with_extras, _ in variants}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            blockers = dict(
                zip(
                    cells,
                    pool.map(
                        lambda cell: _resolve(
                            uv, arguments[cell[0]], cell[1], cell[2], scratch
                        ),
                        cells,
                    ),
                    strict=True,
                )
            )

    notes = []
    for with_extras, title in variants:
        print(f"{title}: wheels only, macOS {MACOS_TARGET}+")
        width = max(len(name) for name in PLATFORMS.values()) + 2
        print((" " * width + "".join(f"{python:<14}" for python in pythons)).rstrip())
        for target, name in PLATFORMS.items():
            row = name.ljust(width)
            for python in pythons:
                blocker = blockers[(with_extras, target, python)]
                expected = _expected(EXPECTED_FAILURES, target, python)
                if with_extras and expected is None:
                    expected = _expected(EXPECTED_EXTRA_FAILURES, target, python)
                on = f"{name}, Python {python}{' with the extras' if with_extras else ''}"
                if blocker is None:
                    cell = "ok" if expected is None else "ok!"
                    if expected is not None:
                        notes.append(f"! {on}: installs now ({expected})")
                else:
                    # The name is shortened, not the mark after it
                    mark = "*" if expected is not None else "!"
                    cell = f"{blocker[:12]}{mark}"
                    if expected is None:
                        notes.append(f"! {on}: no longer installs")
                row += f"{cell:<14}"
            print(row.rstrip())
        print()
    print("* expected:")
    known = [(EXPECTED_FAILURES, "")]
    if extras:
        known.append((EXPECTED_EXTRA_FAILURES, ", with the extras"))
    for failures, suffix in known:
        for (target, python), reason in failures.items():
            on = f"Python {python}" if python else "every Python"
            print(f"  {PLATFORMS[target]}, {on}{suffix}: {reason}")
    for note in notes:
        print(note)
    if notes:
        print(
            "Update EXPECTED_FAILURES or EXPECTED_EXTRA_FAILURES in "
            "dev/check_install.py, and the docs that name the gap."
        )
    return 1 if notes else 0


if __name__ == "__main__":
    sys.exit(main())
