# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Prints the section of a version in the changelog: the notes of its GitHub release.

The publish workflow runs it in its build job, on the tagged commit and before
anything is published, with the version of the tag (``v1.4.1`` gives ``1.4.1``);
once PyPI has the release, its last job creates the GitHub release of the tag,
named after it, with what the script printed. The notes are the text under
``## [<version>] - <date>``, without the heading, up to the next section or
the compare links at the end of the file. A version without a dated section, or
with an empty one, stops the release before it reaches PyPI.

Example:
    uv run python dev/release_notes.py 1.4.1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


_CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def release_notes(changelog: str, version: str) -> str:
    """
    Returns the text of the dated section of a version in the changelog.

    Whole lines only are read: a section heading at the start of a line, and
    the compare links at the end of the file. A mention of a section in the text
    of an entry is neither.

    :param changelog: The text of ``CHANGELOG.md``.
    :type changelog: str
    :param version: The version, e.g. ``1.4.1``.
    :type version: str
    :returns: The text of the section without its heading, ending in a newline.
    :rtype: str
    :raises ValueError: If the changelog has no dated section of the version, or
        the section holds headings only.
    """
    heading = re.search(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
        changelog,
        re.MULTILINE,
    )
    if heading is None:
        raise ValueError(
            f"CHANGELOG.md has no dated section `## [{version}] - <date>`."
        )
    end = re.compile(r"^(?:## \[|\[[^\]]+\]: )", re.MULTILINE).search(
        changelog, heading.end()
    )
    text = changelog[heading.end() : end.start() if end else len(changelog)]
    if not any(line.strip() and not line.startswith("#") for line in text.splitlines()):
        raise ValueError(f"The section [{version}] of CHANGELOG.md is empty.")
    return text.strip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    """
    Prints the notes of the version that the one argument names.

    :param argv: The arguments, e.g. ``["1.4.1"]``. Defaults to those of
        the command line.
    :type argv: list[str] | None
    :returns: 0, or 1 if the arguments or the changelog are not as expected,
        which fails the step that runs the script.
    :rtype: int
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: release_notes.py VERSION, e.g. 1.4.1", file=sys.stderr)
        return 1
    try:
        notes = release_notes(_CHANGELOG.read_text(encoding="utf-8"), args[0])
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    # In UTF-8, whatever the encoding of the console, and with the line ends
    # that reading the file gives (\n)
    sys.stdout.buffer.write(notes.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
