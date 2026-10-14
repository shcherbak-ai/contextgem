# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
File header fixer for pre-commit.

Adds the SPDX lines to the top of each Python file it is given that does not
start with them (after a shebang line, if there is one). Empty files are left
as they are. Exits with 1 when a file was changed, so that the commit stops
and the change can be reviewed and staged.
"""

from __future__ import annotations

import sys
from pathlib import Path


HEADER_LINES = (
    "# SPDX-FileCopyrightText: Shcherbak AI AS",
    "# SPDX-License-Identifier: Apache-2.0",
)


def _add_header(path: Path) -> bool:
    """Add the header to ``path`` if it is missing; return whether it was added."""
    text = path.read_bytes().decode("utf-8")
    if not text.strip():
        return False
    newline = "\r\n" if "\r\n" in text else "\n"
    shebang = ""
    if text.startswith("#!"):
        shebang, _, text = text.partition(newline)
        shebang += newline
    header = newline.join(HEADER_LINES) + newline
    if text.startswith(header):
        return False
    path.write_bytes((shebang + header + newline + text.lstrip("\r\n")).encode("utf-8"))
    return True


def main() -> int:
    """Add the header to the files given on the command line."""
    changed = [name for name in sys.argv[1:] if _add_header(Path(name))]
    for name in changed:
        print(f"Added the SPDX header to {name}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
