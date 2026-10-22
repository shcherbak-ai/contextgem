# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the check of the latest versions and of the advisories
(dev/check_latest_versions.py).
"""

from __future__ import annotations

import datetime
import io
import json
import subprocess  # nosec B404 - a stand-in for the result of uv, nothing is run
import urllib.request
from pathlib import Path

import pytest

from dev import check_latest_versions
from dev.check_latest_versions import (
    ROOT,
    _advisories,
    _CheckError,
    _cooldown,
    _exemptions,
    _local_uv_version,
    _newest,
    _newest_upload,
    _workflow_uv_versions,
)


# The steps of a workflow that sets up uv, after its env
_SETUP_UV = """\
jobs:
  test:
    steps:
      - uses: astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884  # v10.3.0
        with:
          version: ${{ env.UV_VERSION }}
"""


class TestCheckLatestVersions:
    """
    Tests for the check of the latest versions and of the advisories.
    """

    def test_workflow_uv_versions_reads_each_workflow_that_sets_up_uv(
        self, tmp_path: Path
    ):
        """
        Tests that the version of uv is read from each workflow that sets up uv,
        quoted or not, and that a workflow without uv needs none.
        """
        (tmp_path / "a.yml").write_text(
            f'env:\n  UV_VERSION: "0.13.5"\n{_SETUP_UV}', encoding="utf-8"
        )
        (tmp_path / "b.yaml").write_text(
            f"env:\n  UV_VERSION: '0.13.4'  # the fix\n{_SETUP_UV}", encoding="utf-8"
        )
        (tmp_path / "c.yml").write_text(
            f"env:\n  UV_VERSION: 0.13.5\n{_SETUP_UV}", encoding="utf-8"
        )
        (tmp_path / "d.yml").write_text(
            "jobs:\n  test:\n    steps:\n      - uses: actions/setup-python@v7\n",
            encoding="utf-8",
        )
        assert _workflow_uv_versions(tmp_path) == ["0.13.4", "0.13.5"]

    @pytest.mark.parametrize(
        "env",
        [
            "",
            "env:\n  UV_VERSION: ${{ inputs.uv }}\n",
            "env:\n  UV_VERSION: \"0.13.5'\n",
        ],
    )
    def test_workflow_uv_versions_fails_on_uv_without_a_version_to_check(
        self, tmp_path: Path, env: str
    ):
        """
        Tests that a workflow that sets up uv without a literal version fails
        the check, rather than leaving its uv unchecked.
        """
        (tmp_path / "a.yml").write_text(env + _SETUP_UV, encoding="utf-8")
        with pytest.raises(_CheckError, match="a.yml sets up uv without a version"):
            _workflow_uv_versions(tmp_path)

    def test_the_workflows_set_up_uv_of_a_version_to_check(self):
        """
        Tests that the workflows of the repository give the versions of uv that
        they set up, which the check asks osv.dev about.
        """
        assert _workflow_uv_versions(ROOT / ".github" / "workflows")

    def test_advisories_of_uv(self, monkeypatch: pytest.MonkeyPatch):
        """
        Tests that osv.dev is asked about each version of uv, and that an
        advisory of one fails the check, saying where that version is used.
        """
        queries = []

        def answer(request: urllib.request.Request, timeout: float) -> io.BytesIO:
            """
            Answer as osv.dev, with an advisory of each version of uv.
            """
            assert isinstance(request.data, bytes)
            asked = json.loads(request.data)["queries"]
            queries.extend(asked)
            results = [
                {"vulns": [{"id": "GHSA-hq85-5cpj-mr62"}]}
                if query["package"] == {"ecosystem": "PyPI", "name": "uv"}
                else {}
                for query in asked
            ]
            return io.BytesIO(json.dumps({"results": results}).encode("utf-8"))

        monkeypatch.setattr(urllib.request, "urlopen", answer)
        found, seen = _advisories(
            {"package": []},
            [
                ("0.13.1", "UV_VERSION of the workflows"),
                ("0.13.0", "the uv that runs the check"),
            ],
        )
        assert {
            "package": {"ecosystem": "PyPI", "name": "uv"},
            "version": "0.13.1",
        } in queries
        assert {
            "package": {"ecosystem": "PyPI", "name": "uv"},
            "version": "0.13.0",
        } in queries
        assert found == [
            "uv 0.13.1 (PyPI, UV_VERSION of the workflows): GHSA-hq85-5cpj-mr62",
            "uv 0.13.0 (PyPI, the uv that runs the check): GHSA-hq85-5cpj-mr62",
        ]
        assert seen == {"GHSA-hq85-5cpj-mr62"}

    @pytest.mark.parametrize(
        ("output", "version"),
        [
            ("uv 0.13.5 (46776a95d 2026-10-21 x86_64-pc-windows-msvc)\n", "0.13.5"),
            ("uv 0.14.0\n", "0.14.0"),
            ("something else\n", None),
        ],
    )
    def test_local_uv_version_reads_the_output_of_uv(
        self, monkeypatch: pytest.MonkeyPatch, output: str, version: str | None
    ):
        """
        Tests that the version of the uv that runs the check is read from
        `uv --version`, and that an output without one fails the check.
        """

        def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
            """
            Answer as `uv --version`.
            """
            assert command == ["uv", "--version"]
            return subprocess.CompletedProcess(command, 0, output, "")

        monkeypatch.setattr(check_latest_versions, "_run", run)
        if version is None:
            with pytest.raises(_CheckError, match="Cannot read the version of uv"):
                _local_uv_version("uv")
        else:
            assert _local_uv_version("uv") == version

    def test_cooldown_warns_of_an_exemption_until_the_cooldown_admits_it(self):
        """
        Tests that a package taken before the cooldown admits it is a warning
        until a week after its upload, and then fails the check, so that its
        entry is removed; and that an entry without a date fails it.
        """
        table = "pyproject.toml: [tool.uv.exclude-newer-package]"
        pyproject = {
            "tool": {
                "uv": {
                    "exclude-newer": "7 days",
                    "exclude-newer-package": {"Docling_Core": "2026-10-22T14:03:00Z"},
                }
            }
        }
        lock = {"package": []}
        before = datetime.datetime(2026, 10, 29, 14, 2, tzinfo=datetime.UTC)
        exemptions = _exemptions(pyproject)
        assert _cooldown(pyproject, lock, before, exemptions)[1:] == (
            [],
            [
                (
                    f"{table} takes docling-core before the cooldown admits it: "
                    "remove the entry from 2026-10-29 14:03 UTC"
                )
            ],
        )
        after = datetime.datetime(2026, 10, 29, 14, 3, tzinfo=datetime.UTC)
        assert _cooldown(pyproject, lock, after, exemptions)[1:] == (
            [
                (
                    f"{table} docling-core: the cooldown admits it since 2026-10-29 "
                    "14:03 UTC: remove the entry"
                )
            ],
            [],
        )
        pyproject["tool"]["uv"]["exclude-newer-package"] = {"docling": "a week"}
        assert _cooldown(pyproject, lock, after, _exemptions(pyproject))[1] == [
            f"{table} docling = 'a week': date it at its upload"
        ]

    def test_newest_upload_leaves_out_the_exemptions(self):
        """
        Tests that the newest upload of uv.lock, which dates the switch of a
        fixed exclude-newer to the span, leaves out the packages taken before the
        cooldown admits them.
        """
        lock = {
            "package": [
                {
                    "name": "docling-core",
                    "version": "2.104.0",
                    "wheels": [{"upload-time": "2026-10-22T14:02:02Z"}],
                },
                {
                    "name": "pillow",
                    "version": "13.0.0",
                    "sdist": {"upload-time": "2026-10-16T22:15:00Z"},
                    "wheels": [{"upload-time": "2026-10-16T22:16:00Z"}],
                },
            ]
        }
        assert _newest_upload(lock, set()) == datetime.datetime(
            2026, 10, 22, 14, 2, 2, tzinfo=datetime.UTC
        )
        assert _newest_upload(lock, {"docling-core"}) == datetime.datetime(
            2026, 10, 16, 22, 16, tzinfo=datetime.UTC
        )

    def test_newest_passes_the_exemptions_to_uv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """
        Tests that the latest releases are looked up with the packages taken
        before the cooldown admits them, as uv.lock is resolved.
        """
        commands = []

        def run(command: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
            """
            Keep the command, and answer as `uv pip compile`.
            """
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, "docling-core==2.104.0\n")

        monkeypatch.setattr(check_latest_versions, "_run", run)
        newest = _newest(
            "uv",
            ["docling-core"],
            "3.14",
            "7 days",
            {"docling-core": "2026-10-22T14:03:00Z"},
            str(tmp_path),
            "latest",
        )
        assert newest == {"docling-core": "2.104.0"}
        command = commands[0]
        index = command.index("--exclude-newer-package")
        assert command[index + 1] == "docling-core=2026-10-22T14:03:00Z"
