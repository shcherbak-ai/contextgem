# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the changelog and the citation file.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

import contextgem
from dev.date_release import date_release


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
        for version, previous in itertools.pairwise(versions):
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
        # `cz bump` rewrites the version (`version_files` in pyproject.toml), and
        # dev/date_release.py, one of its `pre_bump_hooks`, writes the day into
        # the changelog heading and here. Run as the last of the hooks, this test
        # stops a bump whose two dates differ.
        assert date == released[version]
        # The version of the CFF format, which `cz bump` must leave alone: an
        # unanchored pattern in `version_files` rewrote it in a bump from 1.2.0
        assert re.findall(r"^cff-version: (\S+)$", _CITATION, re.MULTILINE) == ["1.2.0"]

    def test_date_release_names_the_unreleased_section_after_the_version(self):
        """
        Tests that the hook of `cz bump` names the `Unreleased` section after the
        new version with the day, points its compare link at the new tag, and
        writes the day into the citation file, leaving the mentions of the
        section in the text alone; run again after a stopped bump, that it names
        and dates the section again, for another version too.
        """
        mentions = (
            "Changes go under [Unreleased], see `## [Unreleased]`.\n"
            "[Unreleased] entries are dated by `cz bump`.\n\n"
        )
        changelog = (
            "# Changelog\n\n"
            f"{mentions}"
            "## [Unreleased]\n\n### Added\n\n- A change, once in [Unreleased]\n\n"
            f"## [1.1.0] - 2026-10-17\n\n- An earlier change\n\n{mentions}"
            f"[Unreleased]: {_COMPARE}v1.1.0...HEAD\n"
            f"[1.1.0]: {_COMPARE}v1.0.0...v1.1.0\n"
        )
        citation = "version: 1.2.0\ndate-released: 2026-10-17\nurl: x\n"
        dated, cited = date_release(
            changelog, citation, "1.2.0", "v1.2.0", "2026-10-18", released="1.1.0"
        )
        assert dated == (
            "# Changelog\n\n"
            f"{mentions}"
            "## [1.2.0] - 2026-10-18\n\n### Added\n\n- A change, once in [Unreleased]\n\n"
            f"## [1.1.0] - 2026-10-17\n\n- An earlier change\n\n{mentions}"
            f"[1.2.0]: {_COMPARE}v1.1.0...v1.2.0\n"
            f"[1.1.0]: {_COMPARE}v1.0.0...v1.1.0\n"
        )
        assert cited == "version: 1.2.0\ndate-released: 2026-10-18\nurl: x\n"
        again, cited_again = date_release(
            dated, cited, "2.0.0", "v2.0.0", "2026-10-19", released="1.1.0"
        )
        assert again == (
            dated.replace("[1.2.0] - 2026-10-18", "[2.0.0] - 2026-10-19")
            .replace("[1.2.0]:", "[2.0.0]:")
            .replace("...v1.2.0", "...v2.0.0")
        )
        assert cited_again == cited.replace("2026-10-18", "2026-10-19")

    def test_date_release_stops_a_bump_without_unreleased_changes(self):
        """
        Tests that the hook of `cz bump` stops a bump when the newest section of
        the changelog is the one of the version before.
        """
        with pytest.raises(ValueError, match=r"## \[Unreleased\]"):
            date_release(
                _CHANGELOG,
                _CITATION,
                "9.9.9",
                "v9.9.9",
                "2026-10-18",
                released=contextgem.__version__,
            )

    @pytest.mark.parametrize(
        ("sections", "links", "error"),
        [
            # Headings only: nothing to release
            ("## [Unreleased]\n\n### Added\n\n", "", "is empty"),
            # A section twice, e.g. after a merge
            (
                "## [Unreleased]\n\n- A\n\n## [Unreleased]\n\n- B\n\n",
                "",
                "more than one",
            ),
            # A line that only looks like the compare link
            (
                "## [Unreleased]\n\n- A\n\n",
                "[Unreleased]: notes...below\n",
                "no compare link",
            ),
        ],
    )
    def test_date_release_stops_a_bump_on_a_changelog_it_cannot_date(
        self, sections: str, links: str, error: str
    ):
        """
        Tests that the hook of `cz bump` stops a bump, rather than date it, when
        the section to release is empty or there twice, or has no compare link.
        """
        changelog = (
            f"# Changelog\n\n{sections}## [1.1.0] - 2026-10-17\n\n- B\n\n{links}"
            f"[1.1.0]: {_COMPARE}v1.0.0...v1.1.0\n"
        )
        with pytest.raises(ValueError, match=error):
            date_release(
                changelog,
                "date-released: 2026-10-17\n",
                "1.2.0",
                "v1.2.0",
                "2026-10-18",
                released="1.1.0",
            )
