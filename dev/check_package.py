#
# ContextGem
#
# Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Package-build sanity check for pre-commit.

Wipes ``dist/`` so old artifacts don't confuse ``twine check``, runs
``uv build`` to produce a fresh sdist + wheel, then validates them.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - dev script, args are hardcoded lists
import sys
from pathlib import Path


DIST = Path("dist")


def _run(cmd: list[str]) -> int:
    """Run ``cmd`` inheriting stdio; return its exit code."""
    return subprocess.run(cmd).returncode  # nosec B603 - cmd is a hardcoded list, no shell


def main() -> int:
    """Build the package and validate the artifacts."""
    if DIST.exists():
        shutil.rmtree(DIST)

    rc = _run(["uv", "build"])
    if rc != 0:
        return rc

    # Only the actual distributions; skip uv's auto-generated dist/.gitignore.
    artifacts = sorted([*DIST.glob("*.tar.gz"), *DIST.glob("*.whl")])
    if not artifacts:
        print("No artifacts produced in dist/", file=sys.stderr)
        return 1

    return _run(["uv", "run", "twine", "check", *map(str, artifacts)])


if __name__ == "__main__":
    sys.exit(main())
