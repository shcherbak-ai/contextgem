# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Checks that the dependencies are at their latest versions of more than a week
ago, and that no locked version has an advisory.

A release counts only a week after its upload (``COOLDOWN_DAYS``), when a
compromised one has been found and removed: litellm 1.107.0, which stole
credentials, was removed from PyPI two hours after its upload on 2026-10-21. uv
takes the cutoff from ``[tool.uv] exclude-newer`` of ``pyproject.toml`` (the
check passes it to ``uv pip compile``, which runs without the configuration), npm
from ``min-release-age`` of ``.npmrc``. The check fails when either is missing or
is another span. While the newest version of ``uv.lock`` is younger than the
cooldown, ``exclude-newer`` can be a fixed date, past which nothing enters
``uv.lock``: the check warns of it until every version of ``uv.lock`` is older
than the cooldown, and then fails until it is set to the span.

A security fix younger than the cooldown is taken through
``[tool.uv.exclude-newer-package]``, dated at its upload: the check passes the
entries to ``uv pip compile`` too, warns of each until the cooldown admits it, and
then fails until it is removed. Their versions are left out of the newest upload
that dates the switch of a fixed ``exclude-newer`` to the span.

For the Python packages, against PyPI:

- each range of ``pyproject.toml`` (the dependencies, the extras and the
  dependency groups) admits the latest release of its package, on the newest
  Python that ContextGem supports;
- ``uv.lock`` holds the newest versions that the requirements admit, i.e.
  ``uv lock --upgrade`` would change nothing, for the packages that the
  dependencies bring in too. A dependency whose range admits its latest release,
  which the requirement of another package keeps out of the lockfile, is
  reported as a warning, as it cannot be upgraded.

For the development tools of ``package.json``, against npm: each range admits
the latest release, and ``package-lock.json`` holds the newest version that the
range admits. npm reports on the installed packages only, so without npm or
without ``npm ci`` this part is left out, with a note.

The packages in ``EXCEPTIONS`` are left out of every check, with their reasons,
e.g. the dependencies pinned on purpose. A newer release of one is reported as a
warning, which says whether its range admits it. The hook prints the warnings on
a passing commit too (``verbose``). An exception that names no dependency fails
the check.

It asks osv.dev for the advisories of each version of ``uv.lock`` and of
``package-lock.json``: a known vulnerability, or a report of a malicious
package, fails the check, but for the advisories of ``ACCEPTED_ADVISORIES``,
with their reasons. A report of a malicious package (``MAL-``) is never
accepted. An accepted advisory of no locked version fails the check. It asks for
the advisories of uv too, which makes its requests with certificates of its own:
uv 0.13.1 trusted a compromised certificate authority (GHSA-hq85-5cpj-mr62) when
the certifi of ``uv.lock`` was fixed. It asks for the uv that the workflows set up
(``UV_VERSION``), and for the uv that runs the check, with which the dependencies
are locked. A workflow that sets up uv without a version to check fails the check.

Runs as a pre-commit hook at each commit, as a release or an advisory can come
out between two commits, and asks the indexes: uv answers from its cache of
PyPI's pages for up to 10 minutes. CI skips the hook, where its result would
depend on the day rather than on the commit. Offline, or to commit without
upgrading: ``SKIP=latest-versions git commit``.

Example:
    uv run python dev/check_latest_versions.py
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import re
import shutil
import subprocess  # nosec B404 - runs uv and npm with arguments of its own, no shell
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The packages that the check leaves out, by name, with the reasons. A Python
# package is left out of the check of its range and of its locked version, and
# can be one that the dependencies bring in; an npm package is one of
# package.json.
EXCEPTIONS: dict[str, str] = {
    "genai-prices": "pinned, as its API changes often",
    "litellm": "pinned, as its patches and minor versions are at times unstable",
    "openai": "pinned to the version that the pinned litellm is tested with",
}

# How many days a release stays out of the lockfiles after its upload:
# `[tool.uv] exclude-newer` of pyproject.toml and `min-release-age` of .npmrc
COOLDOWN_DAYS = 7

_MARKDOWNLINT_CLI2_0_24 = (
    "markdown-it and smol-toml, which markdownlint-cli2 0.23.3 pins: fixed in "
    "markdownlint-cli2 0.24.0, uploaded on 2026-10-21, which the cooldown keeps out "
    "until 2026-10-28; at worst a slow lint of a Markdown file"
)

# The advisories of locked versions that the check accepts, by their IDs at
# osv.dev, with the reasons: a vulnerability that no version that the ranges
# admit fixes. A report of a malicious package (MAL-) is never accepted
ACCEPTED_ADVISORIES: dict[str, str] = {
    **{
        advisory: _MARKDOWNLINT_CLI2_0_24
        for advisory in (
            "GHSA-gfr2-rv44-r23c",
            "GHSA-r4xh-jqrq-34v2",
            "GHSA-r7fv-28h4-cvq7",
            "GHSA-vfgh-jcmw-g37q",
        )
    },
    "GHSA-238p-pmpm-9mq7": (
        "katex 0.16, which markdownlint's math parser allows (^0.16): fixed in 0.18 "
        "only; markdownlint parses math and renders none"
    ),
}

# The batch query of osv.dev, for the advisories of many versions at once
_OSV_QUERY_BATCH = "https://api.osv.dev/v1/querybatch"

# The version in the output of `uv --version`, e.g. `uv 0.13.5 (46776a95d ...)`
_UV_VERSION_OUTPUT = re.compile(r"uv (\d[\w.+-]*)")

# The version of uv that a workflow sets up, e.g. `UV_VERSION: "0.13.5"`
_UV_VERSION = re.compile(
    r"""^[ \t]*UV_VERSION:[ \t]*(["']?)(\d[\w.+-]*)\1[ \t]*(?:#.*)?$""", re.MULTILINE
)

# The name of a requirement, at its start (PEP 508)
_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# A package and its version in the output of `uv pip compile`
_PINNED = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;]+)")

# A change of the lockfile in the output of `uv lock --dry-run`, e.g.
# `Update sphinx v9.0.4, v9.1.0 -> v9.0.4, v9.1.1`, where the lockfile holds a
# version for the Pythons that the newer one no longer supports
_LOCK_CHANGE = re.compile(r"(Update|Downgrade|Add|Remove) (\S+) (.+)")
_LOCK_VERBS = {"Downgrade": "downgraded", "Add": "added", "Remove": "removed"}


class _CheckError(Exception):
    """The check cannot run: a command failed, or a value cannot be read."""


def _normalize(name: str) -> str:
    """
    Normalize the name of a Python package, as PyPI compares names (PEP 503).

    :param name: E.g. ``Docstring_Parser``.
    :type name: str
    :returns: E.g. ``docstring-parser``.
    :rtype: str
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(pyproject: dict) -> dict[str, list[str]]:
    """
    Collect the requirements of ``pyproject.toml``, by package.

    :param pyproject: The parsed ``pyproject.toml``.
    :type pyproject: dict
    :returns: The requirements of each package, without their markers, by the
        normalized name, e.g. ``{"pillow": ["pillow (>=11.3.0,<14.0.0)"]}``.
    :rtype: dict[str, list[str]]
    """
    project = pyproject["project"]
    groups = [
        project.get("dependencies", []),
        *project.get("optional-dependencies", {}).values(),
        *pyproject.get("dependency-groups", {}).values(),
    ]
    requirements: dict[str, list[str]] = {}
    for group in groups:
        for entry in group:
            # An entry of a dependency group can include another group
            if not isinstance(entry, str):
                continue
            requirement = entry.split(";")[0].strip()
            match = _NAME.match(requirement)
            if match is None:
                raise _CheckError(f"Cannot read the requirement {entry!r}.")
            listed = requirements.setdefault(_normalize(match[1]), [])
            if requirement not in listed:
                listed.append(requirement)
    return requirements


def _newest_python(requires_python: str) -> str:
    """
    Name the newest Python that a ``requires-python`` admits.

    :param requires_python: E.g. ``>=3.11,<3.15``.
    :type requires_python: str
    :returns: E.g. ``3.14``.
    :rtype: str
    """
    match = re.search(r"<\s*3\.(\d+)", requires_python)
    if match is None:
        raise _CheckError(f"Cannot read the newest Python of {requires_python!r}.")
    return f"3.{int(match[1]) - 1}"


def _locked_versions(lock: dict) -> dict[str, list[str]]:
    """
    Read the versions of the packages of ``uv.lock``.

    :param lock: The parsed ``uv.lock``.
    :type lock: dict
    :returns: The versions of each package, by the normalized name: several
        where the lockfile holds one for some Pythons and another for the rest.
    :rtype: dict[str, list[str]]
    """
    locked: dict[str, list[str]] = {}
    for package in lock["package"]:
        if "version" in package:
            locked.setdefault(_normalize(package["name"]), []).append(
                package["version"]
            )
    return locked


def _exemptions(pyproject: dict) -> dict[str, str]:
    """
    Read the packages taken before the cooldown admits them
    (``[tool.uv.exclude-newer-package]``), each with the date of its upload.

    :param pyproject: The parsed ``pyproject.toml``.
    :type pyproject: dict
    :returns: The date of each package, by the normalized name, e.g.
        ``{"docling-core": "2026-10-22T14:03:00Z"}``.
    :rtype: dict[str, str]
    """
    entries = pyproject.get("tool", {}).get("uv", {}).get("exclude-newer-package", {})
    return {_normalize(name): str(date) for name, date in entries.items()}


def _newest_upload(lock: dict, exempt: set[str]) -> datetime.datetime:
    """
    Find when the newest file of ``uv.lock`` was uploaded, but for the packages
    taken before the cooldown admits them.

    :param lock: The parsed ``uv.lock``.
    :type lock: dict
    :param exempt: The normalized names of the packages of
        ``[tool.uv.exclude-newer-package]``.
    :type exempt: set[str]
    :returns: The time of the upload, in UTC.
    :rtype: datetime.datetime
    """
    times = [
        file["upload-time"]
        for package in lock["package"]
        if _normalize(package["name"]) not in exempt
        for file in [*package.get("wheels", []), package.get("sdist", {})]
        if "upload-time" in file
    ]
    if not times:
        raise _CheckError("uv.lock holds no upload times.")
    return max(datetime.datetime.fromisoformat(time) for time in times)


def _cooldown(
    pyproject: dict, lock: dict, now: datetime.datetime, exemptions: dict[str, str]
) -> tuple[str, list[str], list[str]]:
    """
    Check that new releases stay out of the lockfiles for ``COOLDOWN_DAYS``.

    :param pyproject: The parsed ``pyproject.toml``.
    :type pyproject: dict
    :param lock: The parsed ``uv.lock``.
    :type lock: dict
    :param now: The current time, in UTC.
    :type now: datetime.datetime
    :param exemptions: The packages taken before the cooldown admits them, with
        the dates of their uploads (``_exemptions()``).
    :type exemptions: dict[str, str]
    :returns: The cutoff of uv (``exclude-newer``), which the latest releases are
        looked up with; what fails the check; and what it warns of.
    :rtype: tuple[str, list[str], list[str]]
    """
    span = f"{COOLDOWN_DAYS} days"
    findings = []
    warnings = []
    cutoff = pyproject.get("tool", {}).get("uv", {}).get("exclude-newer")
    if cutoff is None:
        findings.append(f'pyproject.toml: set [tool.uv] exclude-newer = "{span}"')
        cutoff = span
    elif re.match(r"\d{4}-\d{2}-\d{2}", cutoff):
        # A fixed date, until the newest version of uv.lock is as old as the
        # cooldown, from when the span changes nothing in uv.lock
        due = _newest_upload(lock, set(exemptions)) + datetime.timedelta(
            days=COOLDOWN_DAYS
        )
        if now >= due:
            findings.append(
                f"pyproject.toml: [tool.uv] exclude-newer is the fixed date {cutoff}, "
                f"and every version of uv.lock is older than {span}: set it to "
                f'"{span}"'
            )
        else:
            warnings.append(
                f"pyproject.toml: [tool.uv] exclude-newer is the fixed date {cutoff}, "
                f'past which nothing enters uv.lock: set it to "{span}" from '
                f"{due:%Y-%m-%d %H:%M} UTC"
            )
    elif cutoff != span:
        findings.append(
            f'pyproject.toml: [tool.uv] exclude-newer is "{cutoff}": set it to "{span}"'
        )
    # A security fix taken before the cooldown admits it, until it does
    table = "pyproject.toml: [tool.uv.exclude-newer-package]"
    for name, date in sorted(exemptions.items()):
        try:
            uploaded = datetime.datetime.fromisoformat(date)
        except ValueError:
            findings.append(f"{table} {name} = {date!r}: date it at its upload")
            continue
        if uploaded.tzinfo is None:
            uploaded = uploaded.replace(tzinfo=datetime.UTC)
        due = uploaded + datetime.timedelta(days=COOLDOWN_DAYS)
        if now >= due:
            findings.append(
                f"{table} {name}: the cooldown admits it since "
                f"{due:%Y-%m-%d %H:%M} UTC: remove the entry"
            )
        else:
            warnings.append(
                f"{table} takes {name} before the cooldown admits it: remove the "
                f"entry from {due:%Y-%m-%d %H:%M} UTC"
            )
    if (ROOT / "package.json").is_file():
        npmrc = ROOT / ".npmrc"
        text = npmrc.read_text(encoding="utf-8") if npmrc.is_file() else ""
        match = re.search(r"^\s*min-release-age\s*=\s*(\d+)\s*$", text, re.MULTILINE)
        if match is None or int(match[1]) != COOLDOWN_DAYS:
            findings.append(f".npmrc: set min-release-age={COOLDOWN_DAYS}")
    return cutoff, findings, warnings


def _workflow_uv_versions(workflows: Path) -> list[str]:
    """
    Read the versions of uv that the workflows set up (``UV_VERSION``).

    :param workflows: The folder of the workflows, ``.github/workflows``.
    :type workflows: Path
    :returns: The versions, e.g. ``["0.13.5"]``.
    :rtype: list[str]
    """
    versions = set()
    for path in sorted(workflows.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        if "astral-sh/setup-uv" not in text:
            continue
        found = [match[2] for match in _UV_VERSION.finditer(text)]
        if not found:
            raise _CheckError(
                f"{path.name} sets up uv without a version to check: set "
                'UV_VERSION: "<version>" in its env.'
            )
        versions.update(found)
    return sorted(versions)


def _local_uv_version(uv: str) -> str:
    """
    Read the version of the uv that runs the check, with which the dependencies
    are locked.

    :param uv: The path of the uv command.
    :type uv: str
    :returns: E.g. ``0.13.5``.
    :rtype: str
    """
    proc = _run([uv, "--version"], cwd=ROOT)
    match = _UV_VERSION_OUTPUT.match(proc.stdout.strip())
    if proc.returncode != 0 or match is None:
        raise _CheckError(
            f"Cannot read the version of uv: {(proc.stdout + proc.stderr).strip()}"
        )
    return match[1]


def _advisories(
    lock: dict, uv_versions: list[tuple[str, str]]
) -> tuple[list[str], set[str]]:
    """
    Ask osv.dev for the advisories of the versions of ``uv.lock``, of
    ``package-lock.json`` and of uv: known vulnerabilities, and reports of
    malicious packages.

    :param lock: The parsed ``uv.lock``.
    :type lock: dict
    :param uv_versions: The versions of uv to ask about, each with where it is
        used, e.g. ``("0.13.5", "UV_VERSION of the workflows")``.
    :type uv_versions: list[tuple[str, str]]
    :returns: Each version with advisories that ``ACCEPTED_ADVISORIES`` does not
        accept, e.g. ``pillow 12.3.0 (PyPI): GHSA-462x-pgg6-55fp``; and the IDs
        of all the advisories found.
    :rtype: tuple[list[str], set[str]]
    """
    packages = sorted(
        {
            ("PyPI", package["name"], package["version"], "")
            for package in lock["package"]
            if "registry" in package.get("source", {})
        }
    )
    npm_lock = ROOT / "package-lock.json"
    if npm_lock.is_file():
        entries = json.loads(npm_lock.read_text(encoding="utf-8"))["packages"]
        packages += sorted(
            {
                ("npm", path.rsplit("node_modules/", 1)[-1], entry["version"], "")
                for path, entry in entries.items()
                if path and "version" in entry and not entry.get("link")
            }
        )
    packages += [
        ("PyPI", "uv", version, f", {where}") for version, where in uv_versions
    ]
    queries = [
        {"package": {"ecosystem": ecosystem, "name": name}, "version": version}
        for ecosystem, name, version, _ in packages
    ]
    request = urllib.request.Request(
        _OSV_QUERY_BATCH,
        data=json.dumps({"queries": queries}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(  # nosec B310 - a fixed https URL of osv.dev
            request, timeout=60
        ) as response:
            results = json.load(response)["results"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        raise _CheckError(f"Cannot ask osv.dev for advisories: {e}") from None
    if len(results) != len(packages):
        raise _CheckError("osv.dev did not answer for each version.")
    found = []
    seen: set[str] = set()
    for (ecosystem, name, version, where), result in zip(
        packages, results, strict=True
    ):
        ids = [vuln["id"] for vuln in result.get("vulns", [])]
        seen.update(ids)
        listed = [
            advisory
            for advisory in ids
            if advisory.startswith("MAL-") or advisory not in ACCEPTED_ADVISORIES
        ]
        # osv.dev sends the advisories of a version in pages
        if result.get("next_page_token"):
            listed.append("and more, at osv.dev")
        if listed:
            found.append(f"{name} {version} ({ecosystem}{where}): {', '.join(listed)}")
    return found, seen


def _run(command: list[str], cwd: Path | str) -> subprocess.CompletedProcess[str]:
    """
    Run a command and capture its output.

    :param command: The command and its arguments.
    :type command: list[str]
    :param cwd: The folder to run it in.
    :type cwd: Path | str
    :returns: The finished process.
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run(  # nosec B603 - uv or npm with arguments of its own, no shell
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        timeout=300,
        check=False,
    )


def _newest(
    uv: str,
    requirements: list[str],
    python: str,
    cutoff: str,
    exemptions: dict[str, str],
    scratch: str,
    label: str,
) -> dict[str, str]:
    """
    Find the newest version that each requirement admits, on its own.

    Without the dependencies of the packages (``--no-deps``), so that no other
    requirement holds a package back, and for any platform (``--universal``).

    :param uv: The path of the uv command.
    :type uv: str
    :param requirements: The requirements, e.g. a bare name for the latest release.
    :type requirements: list[str]
    :param python: The Python version.
    :type python: str
    :param cutoff: The releases uploaded after it are left out (``exclude-newer``
        of uv: a date, or a span back from now, e.g. ``7 days``).
    :type cutoff: str
    :param exemptions: The cutoff of each package taken before the cooldown
        admits it (``exclude-newer-package`` of uv), by the normalized name.
    :type exemptions: dict[str, str]
    :param scratch: A folder to work in, outside the checkout.
    :type scratch: str
    :param label: The name of the requirements file, one per call.
    :type label: str
    :returns: The version of each package, by the normalized name.
    :rtype: dict[str, str]
    """
    path = Path(scratch) / f"{label}.in"
    path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    proc = _run(
        [
            *(uv, "pip", "compile", "--quiet", "--no-config", "--no-header"),
            *("--no-annotate", "--no-deps", "--universal"),
            *("--exclude-newer", cutoff),
            *(
                option
                for name, date in sorted(exemptions.items())
                for option in ("--exclude-newer-package", f"{name}={date}")
            ),
            *("--python-version", python, str(path)),
        ],
        cwd=scratch,
    )
    if proc.returncode != 0:
        raise _CheckError(f"uv pip compile failed:\n{proc.stderr.strip()}")
    return {
        _normalize(match[1]): match[2]
        for line in proc.stdout.splitlines()
        if (match := _PINNED.match(line))
    }


def _lock_changes(uv: str) -> list[tuple[str, str]]:
    """
    List what ``uv lock --upgrade`` would change in ``uv.lock``, without writing it.

    :param uv: The path of the uv command.
    :type uv: str
    :returns: Each change, as the normalized name and a line that describes it,
        e.g. ``("boto3", "boto3 1.43.114 -> 1.43.116")``.
    :rtype: list[tuple[str, str]]
    """
    proc = _run([uv, "lock", "--upgrade", "--dry-run"], cwd=ROOT)
    if proc.returncode != 0:
        raise _CheckError(f"uv lock --upgrade --dry-run failed:\n{proc.stderr}")
    changes = []
    for line in proc.stderr.splitlines():
        match = _LOCK_CHANGE.fullmatch(line.strip())
        if match is None:
            continue
        verb, name, versions = match.groups()
        versions = re.sub(r"\bv(?=\d)", "", versions)
        if verb in _LOCK_VERBS:
            versions += f" ({_LOCK_VERBS[verb]})"
        changes.append((_normalize(name), f"{name} {versions}"))
    # Either uv lists what changes, or it says that nothing does, or that only
    # what holds no version changes, e.g. the order of the markers of the forks
    # of a lockfile that was not resolved afresh: other output would pass the
    # check unread
    if not changes and not re.search(
        r"^(No lockfile changes|Lockfile changes) detected$", proc.stderr, re.MULTILINE
    ):
        raise _CheckError(
            f"Cannot read the output of uv lock --upgrade --dry-run:\n{proc.stderr}"
        )
    return changes


def _npm_outdated() -> tuple[dict[str, str], dict[str, dict] | None, str | None]:
    """
    Ask npm which packages of ``package.json`` are behind their latest releases.

    :returns: The ranges of ``package.json``, by package; the report of ``npm
        outdated`` on the packages that are behind, by package (``wanted``: the
        newest version that the range admits, ``latest``: the latest release),
        or None where npm cannot tell; and then the reason.
    :rtype: tuple[dict[str, str], dict[str, dict] | None, str | None]
    """
    manifest = ROOT / "package.json"
    if not manifest.is_file():
        return {}, {}, None
    package = json.loads(manifest.read_text(encoding="utf-8"))
    ranges = {
        **package.get("dependencies", {}),
        **package.get("devDependencies", {}),
    }
    npm = shutil.which("npm")
    if npm is None:
        return ranges, None, "npm was not found: package.json was not checked."
    # npm reports on the installed packages, and on no other
    missing = [
        name
        for name in ranges
        if not (ROOT / "node_modules" / name / "package.json").is_file()
    ]
    if missing:
        return (
            ranges,
            None,
            (
                f"Not installed: {', '.join(missing)}. Install them (`npm ci`) "
                "to check package.json."
            ),
        )
    # The ranges are not passed as arguments: on Windows, npm runs through
    # cmd.exe, which drops the caret of `^5.9.0`
    proc = _run([npm, "outdated", "--json"], cwd=ROOT)
    # npm exits with 1 when a package is behind, and reports an error in JSON
    try:
        report = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        raise _CheckError(f"npm outdated failed: {proc.stderr.strip()}") from None
    if "error" in report:
        raise _CheckError(f"npm outdated failed: {report['error']}")
    return ranges, report, None


def main() -> int:
    """
    Check the ranges and the lockfiles against the latest releases from before
    the cooldown, and the locked versions and uv against the advisories.

    :returns: 0 when every dependency is at its latest version, or left out, and
        no locked version or uv has an advisory that is not accepted, else 1.
    :rtype: int
    """
    uv = shutil.which("uv")
    if uv is None:
        print("uv was not found.")
        return 1
    # The Python packages left out, by the normalized name
    python_exceptions = {
        _normalize(name): reason for name, reason in EXCEPTIONS.items()
    }
    try:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = _requirements(pyproject)
        python = _newest_python(pyproject["project"]["requires-python"])
        lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
        locked = _locked_versions(lock)
        uv_versions = [
            *(
                (version, "UV_VERSION of the workflows")
                for version in _workflow_uv_versions(ROOT / ".github" / "workflows")
            ),
            (_local_uv_version(uv), "the uv that runs the check"),
        ]
        exemptions = _exemptions(pyproject)
        cutoff, cooldown_findings, cooldown_warnings = _cooldown(
            pyproject, lock, datetime.datetime.now(datetime.UTC), exemptions
        )
        checked = {
            name: listed
            for name, listed in requirements.items()
            if name not in python_exceptions
        }
        # The packages left out too, to warn of their newer releases
        names = sorted(requirements.keys() | (python_exceptions.keys() & locked.keys()))
        ranges = [
            requirement for listed in requirements.values() for requirement in listed
        ]
        with (
            tempfile.TemporaryDirectory(prefix="contextgem-latest-") as scratch,
            concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool,
        ):
            latest_job = pool.submit(
                _newest, uv, names, python, cutoff, exemptions, scratch, "latest"
            )
            admitted_job = pool.submit(
                _newest, uv, ranges, python, cutoff, exemptions, scratch, "ranges"
            )
            changes_job = pool.submit(_lock_changes, uv)
            npm_job = pool.submit(_npm_outdated)
            advisories_job = pool.submit(_advisories, lock, uv_versions)
            latest = latest_job.result()
            admitted = admitted_job.result()
            changes = changes_job.result()
            npm_ranges, npm_report, npm_note = npm_job.result()
            advisories, advisory_ids = advisories_job.result()
        unread = [
            name for name in checked if name not in latest or name not in admitted
        ]
        if unread:
            raise _CheckError(f"uv picked no version of {', '.join(unread)}.")
    except _CheckError as e:
        print(e)
        print("Offline, commit without the check: SKIP=latest-versions git commit")
        return 1

    # The packages left out that have a newer release, which the hook prints on
    # a passing commit too, and whether their ranges admit it
    newer = []
    for name, reason in sorted(python_exceptions.items()):
        release = latest.get(name)
        if name not in locked or release is None or release in locked[name]:
            continue
        where = ""
        if name in requirements:
            within = admitted.get(name) == release
            where = " (within its range)" if within else " (beyond its range)"
        newer.append(f"{name} {', '.join(locked[name])} -> {release}{where}: {reason}")
    # The dependencies whose ranges admit the latest release, which uv.lock does
    # not hold although `uv lock --upgrade` changes nothing: another package's
    # requirement holds them back, which the hook warns of without failing
    changed = {name for name, _ in changes}
    held_back = [
        f"{name} {', '.join(locked[name])} -> {latest[name]}: `uv lock --dry-run "
        f"--upgrade-package {name}=={latest[name]}` says why"
        for name in sorted(checked)
        if admitted[name] == latest[name]
        and name in locked
        and name not in changed
        and latest[name] not in locked[name]
    ]
    npm_ranges_behind = []
    npm_lock_behind = []
    if npm_report is not None:
        lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
        for name, spec in npm_ranges.items():
            entry = npm_report.get(name)
            if entry is None:
                continue
            version = lock["packages"].get(f"node_modules/{name}", {}).get("version")
            if name in EXCEPTIONS:
                within = entry["wanted"] == entry["latest"]
                where = "within" if within else "beyond"
                newer.append(
                    f"{name} {version} -> {entry['latest']} ({where} its range "
                    f"{spec}): {EXCEPTIONS[name]}"
                )
                continue
            if entry["wanted"] != entry["latest"]:
                npm_ranges_behind.append(
                    f"{name} {spec}: latest {entry['latest']}, "
                    f"the range admits up to {entry['wanted']}"
                )
            if version != entry["wanted"]:
                npm_lock_behind.append(f"{name} {version} -> {entry['wanted']}")

    findings = {
        f"The cooldown: a release enters the lockfiles {COOLDOWN_DAYS} days after "
        "its upload": cooldown_findings,
        "Advisories of locked versions and of uv (osv.dev)": advisories,
        "ACCEPTED_ADVISORIES: advisories of no locked version, to remove": sorted(
            ACCEPTED_ADVISORIES.keys() - advisory_ids
        ),
        "pyproject.toml: ranges that leave out the latest release": [
            f"{' and '.join(listed)}: latest {latest[name]}, "
            f"the range admits up to {admitted[name]}"
            for name, listed in sorted(checked.items())
            if admitted[name] != latest[name]
        ],
        "uv.lock: what `uv lock --upgrade` changes": [
            description
            for name, description in changes
            if name not in python_exceptions
        ],
        "package.json: ranges that leave out the latest release": npm_ranges_behind,
        "package-lock.json: what `npm update` changes": npm_lock_behind,
        "EXCEPTIONS: names of no dependency, to remove": [
            name
            for name in EXCEPTIONS
            if _normalize(name) not in locked and name not in npm_ranges
        ],
    }
    for title, lines in findings.items():
        if lines:
            print(title)
            for line in lines:
                print(f"  {line}")
            print()
    if newer:
        print(
            "Warning: newer releases of the packages left out "
            "(EXCEPTIONS in dev/check_latest_versions.py)"
        )
        for line in newer:
            print(f"  {line}")
        print()
    if held_back:
        print("Warning: held back by the requirements of other packages")
        for line in held_back:
            print(f"  {line}")
        print()
    for line in cooldown_warnings:
        print(f"Warning: {line}")
        print()
    if npm_note:
        print(npm_note)
        print()
    if advisories:
        print(
            "An advisory: upgrade past it, or accept it in ACCEPTED_ADVISORIES with "
            "the reason, where no version that the ranges admit fixes it. A "
            "malicious package (MAL-): remove it, and treat each machine that "
            "installed it as compromised: rotate every credential it could reach."
        )
        print()
    if any(findings.values()):
        print(
            "Raise the ranges (up to the next major), upgrade the lockfiles (`uv "
            "lock --upgrade`, `npm update`), or add the package to EXCEPTIONS with "
            "the reason. To commit without upgrading: "
            "SKIP=latest-versions git commit"
        )
        return 1
    files = "pyproject.toml and uv.lock"
    if npm_report is not None:
        files = "pyproject.toml, uv.lock and package.json"
    but = [
        *(["the packages left out"] if EXCEPTIONS else []),
        *(["those held back"] if held_back else []),
    ]
    print(
        f"Every dependency of {files} is at its latest version from before the "
        f"cooldown (exclude-newer: {cutoff})"
        + (f", but for {' and '.join(but)}" if but else "")
        + ", and no locked version or uv has an advisory"
        + (", but for those accepted." if ACCEPTED_ADVISORIES else ".")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
