# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Dates a release in the changelog and the citation file, as a hook of ``cz bump``.

Commitizen rewrites the version fields (``version_files``), and runs this script
among its ``pre_bump_hooks``, with the new version in ``CZ_PRE_NEW_VERSION``, its
tag in ``CZ_PRE_NEW_TAG_VERSION`` and the version before it in
``CZ_PRE_CURRENT_VERSION``. The script renames the ``## [Unreleased]`` section of
``CHANGELOG.md`` to the new version with the day of the bump (the local date),
points the section's compare link at the new tag instead of ``HEAD``, and writes
the day into ``date-released`` of ``CITATION.cff``. The bump commits with
``git commit -a``, which takes the two files in.

Any newest section but the one of the version before is taken as the release:
also one that a bump stopped by a later hook has named, which is named and dated
again on the next try. Without such a section, the script stops the bump, and
changes nothing.

Example:
    uv run cz bump
"""

from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_CHANGELOG = _ROOT / "CHANGELOG.md"
_CITATION = _ROOT / "CITATION.cff"


def date_release(
    changelog: str, citation: str, version: str, tag: str, day: str, released: str
) -> tuple[str, str]:
    """
    Date a release in the texts of the changelog and of the citation file.

    :param changelog: The text of ``CHANGELOG.md``.
    :type changelog: str
    :param citation: The text of ``CITATION.cff``.
    :type citation: str
    :param version: The new version, e.g. ``1.2.0``.
    :type version: str
    :param tag: The tag of the new version, e.g. ``v1.2.0``.
    :type tag: str
    :param day: The release date, e.g. ``2026-10-18``.
    :type day: str
    :param released: The version before the new one, e.g. ``1.1.0``.
    :type released: str
    :returns: The two texts, dated.
    :rtype: tuple[str, str]
    :raises ValueError: If the newest section of the changelog is the one of the
        version before, or it is empty, or it has no compare link, or the
        changelog has two ``Unreleased`` sections, or the citation file has no
        ``date-released``.
    """
    # Only whole lines are read and changed: a section heading (`## [...]` at the
    # start of a line), the section's compare link (`[...]: <URL>` at the start
    # of a line) and `date-released`. A mention of a section in the text of an
    # entry is none of them.
    if len(re.findall(r"^## \[Unreleased\]", changelog, re.MULTILINE)) > 1:
        raise ValueError("CHANGELOG.md has more than one `## [Unreleased]` section.")
    # The newest section: `Unreleased`, or one that a stopped bump has named
    newest = re.search(r"^## \[(.+?)\].*$", changelog, re.MULTILINE)
    if newest is None or newest[1] == released:
        raise ValueError(
            f"CHANGELOG.md has no changes above the section of {released} to "
            f"release as {version}: write them under `## [Unreleased]`."
        )
    # Its text runs to the next section, or to the compare links at the end
    end = re.compile(r"^(?:## \[|\[[^\]]+\]: )", re.MULTILINE).search(
        changelog, newest.end()
    )
    text = changelog[newest.end() : end.start() if end else len(changelog)]
    if not any(line.strip() and not line.startswith("#") for line in text.splitlines()):
        raise ValueError(
            f"The section [{newest[1]}] of CHANGELOG.md is empty: write the changes "
            "of the release in it."
        )
    name = re.escape(newest[1])
    changelog = (
        changelog[: newest.start()]
        + f"## [{version}] - {day}"
        + changelog[newest.end() :]
    )
    changelog, links = re.subn(
        rf"^\[{name}\]: (\S+/compare/\S+?)\.\.\.\S+$",
        lambda link: f"[{version}]: {link[1]}...{tag}",
        changelog,
        count=1,
        flags=re.MULTILINE,
    )
    if not links:
        raise ValueError(f"CHANGELOG.md has no compare link for [{newest[1]}].")
    citation, dates = re.subn(
        r"^date-released: .*$",
        f"date-released: {day}",
        citation,
        count=1,
        flags=re.MULTILINE,
    )
    if not dates:
        raise ValueError("CITATION.cff has no `date-released`.")
    return changelog, citation


def main() -> int:
    """
    Date the release that ``cz bump`` makes.

    :returns: 0 when the release is dated, else 1, which stops the bump.
    :rtype: int
    """
    version = os.environ.get("CZ_PRE_NEW_VERSION")
    released = os.environ.get("CZ_PRE_CURRENT_VERSION")
    if not version or not released:
        print(
            "Run it as a hook of `cz bump`, which sets CZ_PRE_NEW_VERSION and "
            "CZ_PRE_CURRENT_VERSION.",
            file=sys.stderr,
        )
        return 1
    tag = os.environ.get("CZ_PRE_NEW_TAG_VERSION") or f"v{version}"
    day = date.today().isoformat()
    try:
        changelog, citation = date_release(
            _CHANGELOG.read_text(encoding="utf-8"),
            _CITATION.read_text(encoding="utf-8"),
            version,
            tag,
            day,
            released,
        )
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    # Both files are written once both are dated, with the line endings of the
    # repository
    _CHANGELOG.write_text(changelog, encoding="utf-8", newline="\n")
    _CITATION.write_text(citation, encoding="utf-8", newline="\n")
    print(f"Dated {version}: {day} (CHANGELOG.md, CITATION.cff)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
