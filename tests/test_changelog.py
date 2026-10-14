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
Module defining tests for the changelog and the citation file.
"""

from __future__ import annotations

import re
from pathlib import Path

import contextgem


_ROOT = Path(__file__).resolve().parent.parent
_CHANGELOG = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
_CITATION = (_ROOT / "CITATION.cff").read_text(encoding="utf-8")
_COMPARE = "https://github.com/shcherbak-ai/contextgem/compare/"


def _ref(version: str) -> str:
    """
    Returns what a comparison ends at: the tag of the version, or ``HEAD`` for
    unreleased work.

    :param version: Version of a section of the changelog
    :type version: str
    :returns: The ref that the comparison link of the section ends at
    :rtype: str
    """
    return "HEAD" if version == "Unreleased" else f"v{version}"


class TestChangelog:
    """
    Tests for the changelog and the citation file.
    """

    def test_each_version_links_to_its_comparison_with_the_one_before(self):
        """
        Tests that each version of the changelog has a compare link with the
        version before it. A version without its link raises no error anywhere:
        its heading just keeps the brackets.
        """
        versions = re.findall(r"^## \[(.+?)\]", _CHANGELOG, re.MULTILINE)
        links = dict(re.findall(r"^\[(.+?)\]: (\S+)$", _CHANGELOG, re.MULTILINE))
        # The links stand at the end of the file, newest first like the
        # sections: a new version's link goes on top.
        assert list(links) == versions
        for version, previous in zip(versions, versions[1:], strict=False):
            assert links[version] == f"{_COMPARE}v{previous}...{_ref(version)}"
        # The oldest section follows the 0.1.x releases, which the changelog
        # does not list.
        oldest = versions[-1]
        assert re.fullmatch(
            rf"{re.escape(_COMPARE)}v[\d.]+\.\.\.v{re.escape(oldest)}", links[oldest]
        )

    def test_citation_file_names_the_package_version_and_its_release_date(self):
        """
        Tests that the citation file names the version of the package and the
        release date that the changelog gives for that version.
        """
        released = dict(
            re.findall(
                r"^## \[(.+?)\] - (\d{4}-\d{2}-\d{2})$", _CHANGELOG, re.MULTILINE
            )
        )
        (version,) = re.findall(r"^version: (\S+)$", _CITATION, re.MULTILINE)
        (date,) = re.findall(r"^date-released: (\S+)$", _CITATION, re.MULTILINE)
        assert version == contextgem.__version__
        # `cz bump` rewrites the version (`version_files` in pyproject.toml) and
        # leaves the date: the release change writes the day into the changelog
        # heading and here. Run as the last of the `pre_bump_hooks`, this test
        # stops a bump that forgot either.
        assert date == released[version]
