# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Checks the links of the README, the package description, the changelog, the
documentation and the package.

By default, the links to ContextGem's own documentation site and repository,
offline, against what the next release publishes: the site is built from main,
the release branch, so a page or a heading new on dev is missing there until the
release, and a link to it gives 404 until then.

- A link to the documentation site (``SITE``, or ``OLD_SITE``, which redirects
  there with the path) leads to a page or a file of the local build of the
  documentation, and its anchor to an element of the page.
- A link to a file of the repository on main (``/blob/main/``, ``/tree/main/``,
  ``/raw/main/``) leads to a file of the working tree, and its anchor in a
  Markdown file to a heading, by GitHub's rules, or to an element with that id. A
  link to a workflow leads to its file, and a compare or release link names tags
  that exist, or the tag of the version of ``pyproject.toml``, which the bump that
  makes the release creates.
- A relative link of a Markdown file leads to a file, and its anchor to a heading.

The links that the releases published are checked the same way: those of their
package descriptions (``project.readme``), their project URLs, and the docstrings
and messages of their package, which stay on PyPI and in the installed copies. A
page or a file that one of them links to is thus not renamed or deleted. This
needs the tags of the repository. The releases from ``OLDEST_CHECKED_RELEASE`` on
are checked: those before it link to pages and anchors that the redesign of 1.0
moved or removed, e.g. the pages ending in ``.html``.

With ``--external``, the links to other sites, online: a link fails when its
site answers that the page is not there (404 or 410), or its host does not
exist. A site that refuses the request (e.g. 403 to a script), limits the rate
(429) or does not answer is reported without failing the check, as the page may
be there. ``IGNORED`` leaves out the links that lead nowhere on purpose, e.g.
placeholders, with the reasons; an entry that matches no link fails the check.
The links to hosts that cannot be public, e.g. ``localhost``, are left out,
and so is a host that resolves to an address that is not public, e.g. one of
a local network or a cloud's metadata service, and a redirect to one, or to
another scheme than http and https: the files can come from a contributor,
and the requests from a maintainer's machine. A request is a GET of the link,
without its anchor, with fixed headers: no body, cookies or credentials. Only
the status of an answer is read, never its body. The check waits at most
``_DEADLINE`` for all the answers: a link without an answer by then is not
checked, so that a site that hangs cannot hold the job.

CI runs the default check after the strict build of the documentation, on each
push and pull request (``.github/workflows/docs.yml``). The external check
depends on other sites, whose answers change without a commit, so CI runs it
weekly (``.github/workflows/links.yml``). Before a release, ``uv run fab links``
builds the documentation and runs both. Neither sends a request to a model.

Example:
    uv run python dev/check_links.py
    uv run python dev/check_links.py --build docs/_build/dirhtml
    uv run python dev/check_links.py --external
"""

from __future__ import annotations

import argparse
import collections
import html
import ipaddress
import queue
import re
import socket
import ssl
import subprocess  # nosec B404 - runs git with arguments of its own, no shell
import sys
import threading
import time
import tomllib
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO


ROOT = Path(__file__).resolve().parents[1]

# The documentation site, and its old domain, which redirects there with the path
SITE = "https://contextgem.shcherbak.ai/"
OLD_SITE = "https://contextgem.dev/"

REPOSITORY = "https://github.com/shcherbak-ai/contextgem"
_RAW_REPOSITORY = "https://raw.githubusercontent.com/shcherbak-ai/contextgem/"

# The oldest release whose links are checked: the releases before it link to
# pages and anchors that the redesign of 1.0 moved or removed
OLDEST_CHECKED_RELEASE = "v1.0.0"

# Where `uv run fab docs` builds the documentation; CI builds into
# docs/_build/dirhtml and passes it with --build
DEFAULT_BUILD = ROOT / "docs" / "build" / "dirhtml"

# The files whose links are checked, as glob patterns from the root: what the
# readers of the repository, of PyPI, of the documentation and of the package
# see. Left out: LICENSES.md, made from the metadata of the dependencies by
# licenseal; dev/readme.template.md, whose links README.md has; the recorded
# results of dev/evals, which are never edited; and the signed agreements of
# .github/contributors.
SOURCES = (
    "*.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "pyproject.toml",
    ".github/*.md",
    ".github/ISSUE_TEMPLATE/*",
    "dev/evals/AGENTS.md",
    "docs/source/**/*.rst",
    "docs/source/conf.py",
    "docs/source/robots.txt",
    "contextgem/**/*.py",
    "dev/usage_examples/**/*.py",
)
_EXCLUDED_SOURCES = {"LICENSES.md"}

# The links to other sites that lead nowhere on purpose, by the start of the
# link, with the reasons
IGNORED: dict[str, str] = {
    "https://github.com/YOUR-GITHUB-USERNAME/": (
        "a placeholder in the steps of CONTRIBUTING.md: the fork of the reader"
    ),
}

# Hosts that cannot be public: local servers, and the names reserved for
# examples and tests (RFC 2606, RFC 6761)
_LOCAL_HOST = re.compile(
    r"(?:localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1"
    r"|(?:.+\.)?(?:local|localhost|example|test|invalid)"
    r"|(?:.+\.)?example\.(?:com|net|org))",
    re.IGNORECASE,
)

# A link in any text: up to a space, a quote or a bracket. Its end is cut from
# punctuation that ends the sentence or the markup around it.
_URL = re.compile(r"https?://[^\s<>\"'`()\[\]{}|\\^]+")
_URL_END = ".,;:!?*_"

# The links of Markdown: [text](target), [name]: target, and the attributes of
# HTML elements
_MARKDOWN_TARGETS = (
    re.compile(r"\]\(\s*<?([^()\s<>]+)>?(?:\s+\"[^\"]*\")?\s*\)"),
    re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)"),
    re.compile(r"\b(?:href|src)=[\"']([^\"']+)[\"']"),
)
_CODE_SPAN = re.compile(r"(`+).+?\1")
# Inline code of reST: in single backticks is a link, e.g. `text <address>`_
_RST_LITERAL = re.compile(r"``.+?``")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

# How a site is asked for a page: as a browser asks. 13 of the 77 other sites
# that the files linked to refused urllib's own agent (403, or 999 from
# LinkedIn), and none refused this one (2026-10-22)
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0 Safari/537.36"
)
_TIMEOUT = 30
_ATTEMPTS = 3
_WORKERS = 8
# The longest wait that a site's Retry-After is followed for, in seconds
_MAX_RETRY_AFTER = 60
# The longest the external check waits for all the answers, in seconds, well
# within the 15 minutes of its job (links.yml): the timeout above holds for each
# read, so a site that sends its answer a byte at a time would hold a request
# for longer
_DEADLINE = 480


class _CheckError(Exception):
    """
    The check cannot run, e.g. without the build of the documentation.
    """


@dataclass(frozen=True)
class Link:
    """
    A link, where it was found.

    :ivar url: The link, as written.
    :vartype url: str
    :ivar source: The file it was found in, from the root of the repository, or
        with the tag of the release before it, e.g. ``v1.1.0:PYPI.md``.
    :vartype source: str
    :ivar line: The number of its line, from 1.
    :vartype line: int
    """

    url: str
    source: str
    line: int

    def __str__(self) -> str:
        """
        Return where the link is, and the link.

        :returns: E.g. ``README.md:12: https://...``.
        :rtype: str
        """
        return f"{self.source}:{self.line}: {self.url}"


def github_anchor(heading: str) -> str:
    """
    Return the anchor that GitHub gives a heading of a Markdown file: its text in
    lower case, with the letters, marks, numbers, connectors (``_``), hyphens and
    spaces kept, and each space a hyphen. An emoji at the start thus leaves a
    hyphen, e.g. ``-installation`` for ``📦 Installation``.

    :param heading: The text of the heading, without the ``#`` marks.
    :type heading: str
    :returns: The anchor, without ``#``.
    :rtype: str
    """
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]*>", "", text).strip().lower()
    kept = "".join(
        character
        for character in text
        if character in "- "
        or unicodedata.category(character)[0] in "LMN"
        or unicodedata.category(character) == "Pc"
    )
    return kept.replace(" ", "-")


def markdown_anchors(text: str) -> set[str]:
    """
    Return the anchors of a Markdown file as GitHub shows it: those of its
    headings, a repeated one with ``-1``, ``-2``... after it, and the ids and
    names of its HTML elements.

    :param text: The Markdown file.
    :type text: str
    :returns: The anchors, without ``#``.
    :rtype: set[str]
    """
    anchors: set[str] = set()
    counts: collections.Counter[str] = collections.Counter()
    for line in _outside_code_blocks(text):
        heading = re.match(r" {0,3}#{1,6}\s+(.*?)(?:\s+#+)?\s*$", line)
        if heading:
            anchor = github_anchor(heading.group(1))
            repeated = counts[anchor]
            counts[anchor] += 1
            anchors.add(f"{anchor}-{repeated}" if repeated else anchor)
        anchors.update(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", line))
    return anchors


def html_ids(text: str) -> set[str]:
    """
    Return the ids of the elements of an HTML page, which its anchors lead to.

    :param text: The HTML page.
    :type text: str
    :returns: The ids.
    :rtype: set[str]
    """
    return {html.unescape(found) for found in re.findall(r"\bid=\"([^\"]+)\"", text)}


def _outside_code_blocks(text: str) -> Iterator[str]:
    """
    Yield the lines of a Markdown file, each line of a fenced code block as an
    empty line, so that the lines keep their numbers.

    :param text: The Markdown file.
    :type text: str
    :returns: The lines.
    :rtype: Iterator[str]
    """
    fence = ""
    for line in text.splitlines():
        marker = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if fence:
            if marker and marker.group(1)[0] == fence[0] and marker.group(1) >= fence:
                fence = ""
            yield ""
        elif marker:
            fence = marker.group(1)
            yield ""
        else:
            yield line


def find_urls(source: str, text: str) -> Iterator[Link]:
    """
    Yield the absolute links (http and https) of a text. A link followed by a
    ``{`` or a ``<`` is a placeholder of a template, e.g.
    ``https://{resource}.openai.azure.com``, and is left out. So is an address in
    inline code of a Markdown or reST file, which is a value rather than a link,
    e.g. an API base or a prefix of addresses, and which neither makes a link of;
    a code block keeps its addresses, e.g. those of a command.

    :param source: The name of the file, for the links.
    :type source: str
    :param text: The text of the file.
    :type text: str
    :returns: The links.
    :rtype: Iterator[Link]
    """
    code = {".md": _CODE_SPAN, ".rst": _RST_LITERAL}.get(Path(source).suffix.lower())
    for number, line in enumerate(text.splitlines(), start=1):
        if code:
            line = code.sub("", line)
        for match in _URL.finditer(line):
            following = line[match.end() : match.end() + 1]
            url = match.group().rstrip(_URL_END)
            if following and following in "{<":
                continue
            if not urllib.parse.urlsplit(url).hostname:
                continue
            yield Link(url, source, number)


def find_relative_links(source: str, text: str) -> Iterator[Link]:
    """
    Yield the relative links of a Markdown file, e.g. ``CONTRIBUTING.md``,
    ``docs/source/_static/figure.svg`` or ``#installation``, outside its code.

    :param source: The name of the file, for the links.
    :type source: str
    :param text: The Markdown file.
    :type text: str
    :returns: The links.
    :rtype: Iterator[Link]
    """
    for number, line in enumerate(_outside_code_blocks(text), start=1):
        line = _CODE_SPAN.sub("", line)
        for pattern in _MARKDOWN_TARGETS:
            for target in pattern.findall(line):
                if _SCHEME.match(target) or target.startswith("//"):
                    continue
                yield Link(target, source, number)


def _source_files(root: Path) -> list[Path]:
    """
    Return the files whose links are checked (``SOURCES``).

    :param root: The root of the repository.
    :type root: Path
    :returns: The files, sorted.
    :rtype: list[Path]
    """
    files = {
        path
        for pattern in SOURCES
        for path in root.glob(pattern)
        if path.is_file() and path.name not in _EXCLUDED_SOURCES
    }
    return sorted(files)


def _git(*arguments: str) -> str:
    """
    Run git in the repository and return what it prints.

    :param arguments: The arguments of git.
    :type arguments: str
    :returns: The output.
    :rtype: str
    :raises _CheckError: If git fails.
    """
    try:
        result = subprocess.run(  # nosec B603 B607 - git, with arguments of its own
            ["git", *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            timeout=120,
            check=False,
        )
    except OSError as e:
        raise _CheckError(f"Cannot run git: {e}") from None
    if result.returncode not in (0, 1):
        raise _CheckError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


class OwnLinks:
    """
    Checks the links to the documentation site and to the repository against
    what the next release publishes: the build of the documentation, and the
    working tree.

    :ivar build: The build of the documentation (dirhtml).
    :vartype build: Path
    :ivar root: The working tree.
    :vartype root: Path
    :ivar refs: The refs that a compare or release link may name: the tags, the
        tag of the version of ``pyproject.toml``, ``HEAD`` and the branches.
    :vartype refs: set[str]
    """

    def __init__(self, build: Path, root: Path, refs: set[str]) -> None:
        """
        :param build: The build of the documentation (dirhtml).
        :type build: Path
        :param root: The working tree.
        :type root: Path
        :param refs: The refs that a compare or release link may name.
        :type refs: set[str]
        """
        self.build = build
        self.root = root
        self.refs = refs
        self._anchors: dict[Path, set[str]] = {}

    def problem(self, url: str) -> str | None:
        """
        Return why a link that ``is_own()`` takes leads nowhere.

        :param url: The link.
        :type url: str
        :returns: The reason, or ``None`` if the link leads somewhere.
        :rtype: str | None
        """
        address, _, fragment = url.partition("#")
        address = address.split("?", 1)[0]
        fragment = urllib.parse.unquote(fragment)
        for site in (SITE, OLD_SITE):
            if address.rstrip("/") + "/" == site or address.startswith(site):
                return self._site_problem(address[len(site) :], fragment)
        if address.startswith(_RAW_REPOSITORY):
            path = address[len(_RAW_REPOSITORY) :].removeprefix("refs/heads/")
            if not path.startswith("main/"):
                return "not a link to main"
            return self._file_problem(path.removeprefix("main/"), "", file=True)
        path = _repository_path(address) or ""
        if found := re.match(r"/(blob|tree|raw)/main/(.+)", path):
            kind, name = found.groups()
            return self._file_problem(name, fragment, file=kind != "tree")
        if found := re.fullmatch(r"/compare/(.+?)\.{2,3}(.+)", path):
            missing = [ref for ref in found.groups() if ref not in self.refs]
            return f"no tag {', '.join(missing)}" if missing else None
        if found := re.fullmatch(r"/releases/tag/(.+)", path):
            return None if found.group(1) in self.refs else "no such tag"
        if found := re.fullmatch(r"/actions/workflows/([^/]+)(?:/badge\.svg)?", path):
            workflow = self.root / ".github" / "workflows" / found.group(1)
            return None if workflow.is_file() else "no such workflow"
        return "not a link that this check takes"

    def _site_problem(self, path: str, fragment: str) -> str | None:
        """
        Return why a page of the site, or its anchor, is missing in the build.

        :param path: The path of the page on the site, e.g. ``documents/docling/``.
        :type path: str
        :param fragment: The anchor, or an empty string.
        :type fragment: str
        :returns: The reason, or ``None`` if the page and the anchor are there.
        :rtype: str | None
        """
        target = self.build / urllib.parse.unquote(path)
        if not path or path.endswith("/") or target.is_dir():
            target = target / "index.html"
        if not _inside(target, self.build):
            return "leads out of the build of the documentation"
        if not target.is_file():
            return "no such page in the build of the documentation"
        is_page = target.suffix == ".html"
        if fragment and is_page and fragment not in self._ids(target, html_ids):
            return f"no anchor #{fragment} on the page"
        return None

    def _file_problem(self, name: str, fragment: str, file: bool) -> str | None:
        """
        Return why a file of the repository, or its anchor, is missing in the
        working tree.

        :param name: The path of the file in the repository.
        :type name: str
        :param fragment: The anchor, or an empty string.
        :type fragment: str
        :param file: Whether the link is to a file, rather than to a folder.
        :type file: bool
        :returns: The reason, or ``None`` if the file and the anchor are there.
        :rtype: str | None
        """
        target = self.root / urllib.parse.unquote(name)
        if not _inside(target, self.root):
            return "leads out of the repository"
        if not (target.is_file() if file else target.exists()):
            return "no such file in the working tree"
        return self._anchor_problem(target, fragment)

    def _anchor_problem(self, target: Path, fragment: str) -> str | None:
        """
        Return why the anchor of a link to a file of the repository is missing:
        a heading of a Markdown file, or a line of another file (``#L12``).

        :param target: The file.
        :type target: Path
        :param fragment: The anchor, or an empty string.
        :type fragment: str
        :returns: The reason, or ``None`` if the anchor is there.
        :rtype: str | None
        """
        if not fragment or not target.is_file():
            return None
        if lines := re.fullmatch(r"L(\d+)(?:-L(\d+))?", fragment):
            count = len(target.read_text(encoding="utf-8").splitlines())
            last = max(int(number) for number in lines.groups() if number)
            return None if last <= count else f"no line {last} in the file"
        is_markdown = target.suffix.lower() == ".md"
        if is_markdown and fragment not in self._ids(target, markdown_anchors):
            return f"no heading or element with the anchor #{fragment}"
        return None

    def _ids(self, target: Path, reader: Callable[[str], set[str]]) -> set[str]:
        """
        Return the anchors of a file, read once.

        :param target: The file.
        :type target: Path
        :param reader: What reads them from its text: ``html_ids()`` or
            ``markdown_anchors()``.
        :type reader: Callable[[str], set[str]]
        :returns: The anchors.
        :rtype: set[str]
        """
        if target not in self._anchors:
            self._anchors[target] = reader(target.read_text(encoding="utf-8"))
        return self._anchors[target]

    def relative_problem(self, link: Link, source: Path) -> str | None:
        """
        Return why a relative link of a Markdown file leads nowhere.

        :param link: The link.
        :type link: Link
        :param source: The Markdown file.
        :type source: Path
        :returns: The reason, or ``None`` if the link leads somewhere.
        :rtype: str | None
        """
        address, _, fragment = link.url.partition("#")
        address = urllib.parse.unquote(address.split("?", 1)[0])
        fragment = urllib.parse.unquote(fragment)
        if not address:
            target = source
        elif address.startswith("/"):
            target = self.root / address.lstrip("/")
        else:
            target = source.parent / address
        if not _inside(target, self.root):
            return "leads out of the repository"
        if not target.exists():
            return "no such file in the working tree"
        return self._anchor_problem(target, fragment)


def is_own(url: str) -> bool:
    """
    Return whether the offline check takes a link: one to the documentation
    site, or to a file, a workflow, a comparison or a release of the
    repository. The other links of the repository, e.g. its issues, are external.

    :param url: The link.
    :type url: str
    :returns: Whether the offline check takes it.
    :rtype: bool
    """
    if url.startswith((SITE, OLD_SITE, _RAW_REPOSITORY)):
        return True
    if url.rstrip("/") + "/" in (SITE, OLD_SITE):
        return True
    path = _repository_path(url)
    return path is not None and bool(
        re.match(r"/(?:(?:blob|tree|raw)/main/|compare/|releases/tag/)", path)
        or re.fullmatch(r"/actions/workflows/[^/]+(?:/badge\.svg)?", path)
    )


def _inside(target: Path, folder: Path) -> bool:
    """
    Return whether a path stays inside a folder once resolved: a link's path
    with ``..``, or an absolute one, which a join keeps as it is, would lead the
    check to the other files of the machine.

    :param target: The path.
    :type target: Path
    :param folder: The folder.
    :type folder: Path
    :returns: Whether the path is inside the folder.
    :rtype: bool
    """
    return target.resolve().is_relative_to(folder.resolve())


def _repository_path(url: str) -> str | None:
    """
    Return the path of a link to the repository after its address, e.g.
    ``/blob/main/CHANGELOG.md``.

    :param url: The link.
    :type url: str
    :returns: The path, an empty string for the repository itself, or ``None``
        for a link elsewhere.
    :rtype: str | None
    """
    if not url.startswith(REPOSITORY):
        return None
    rest = url[len(REPOSITORY) :]
    if rest and rest[0] not in "/#?.":
        return None
    return rest.split("#", 1)[0].split("?", 1)[0]


def _refs(root_version: str) -> set[str]:
    """
    Return the refs that a compare or release link may name: the tags, the tag
    of the version being released, ``HEAD`` and the branches.

    :param root_version: The version of ``pyproject.toml``.
    :type root_version: str
    :returns: The refs.
    :rtype: set[str]
    """
    tags = set(_git("tag", "--list").split())
    return tags | {f"v{root_version}", "HEAD", "main", "dev"}


def _release_tags() -> list[str]:
    """
    Return the tags of the releases whose links are checked: from
    ``OLDEST_CHECKED_RELEASE`` to the latest.

    :returns: The tags, the latest first, e.g. ``["v1.1.0", "v1.0.0"]``.
    :rtype: list[str]
    :raises _CheckError: If the repository does not have the oldest of them, e.g.
        in a clone without its tags.
    """
    tags = _git("tag", "--list", "v*", "--sort=-v:refname").split()
    if OLDEST_CHECKED_RELEASE not in tags:
        raise _CheckError(
            f"The repository has no tag {OLDEST_CHECKED_RELEASE}, so the links of "
            "the releases cannot be checked: fetch the tags (git fetch --tags), or "
            "in CI check out with fetch-depth: 0."
        )
    return tags[: tags.index(OLDEST_CHECKED_RELEASE) + 1]


def released_links(tag: str) -> list[Link]:
    """
    Return the links that a release published, which stay on PyPI and in the
    installed copies: those of its package description (``project.readme``), of
    its ``pyproject.toml`` (the project URLs) and of its package.

    :param tag: The tag of the release.
    :type tag: str
    :returns: The links, with sources such as ``v1.1.0:PYPI.md``.
    :rtype: list[Link]
    """
    pyproject = tomllib.loads(_git("show", f"{tag}:pyproject.toml"))
    readme = pyproject["project"].get("readme", "README.md")
    if isinstance(readme, dict):
        readme = readme["file"]
    paths = [readme, "pyproject.toml", "contextgem"]
    output = _git("grep", "-n", "-I", "-E", "https?://", tag, "--", *paths)
    links = []
    for row in output.splitlines():
        _, name, number, text = row.split(":", 3)
        links += [
            Link(link.url, f"{tag}:{name}", int(number))
            for link in find_urls(name, text)
        ]
    return links


def check_own(build: Path) -> int:
    """
    Check the links to the documentation site and to the repository, and the
    relative links of the Markdown files, of the working tree and of the
    releases, and print what leads nowhere.

    :param build: The build of the documentation (dirhtml).
    :type build: Path
    :returns: The exit code: 0 if every link leads somewhere, else 1.
    :rtype: int
    :raises _CheckError: If the build or the tags are missing.
    """
    if not (build / "index.html").is_file():
        raise _CheckError(
            f"No build of the documentation in {build}: build it first "
            "(uv run fab docs), or give its folder with --build."
        )
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    checker = OwnLinks(build, ROOT, _refs(version))
    broken: list[str] = []
    counts: collections.Counter[str] = collections.Counter()
    for path in _source_files(ROOT):
        name = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for link in find_urls(name, text):
            if is_own(link.url):
                counts["own"] += 1
                if reason := checker.problem(link.url):
                    broken.append(f"{link}: {reason}")
        if path.suffix.lower() == ".md":
            for link in find_relative_links(name, text):
                counts["relative"] += 1
                if reason := checker.relative_problem(link, path):
                    broken.append(f"{link}: {reason}")
    tags = _release_tags()
    # A link that several releases have is checked once, and named with each
    released: dict[str, list[Link]] = collections.defaultdict(list)
    for tag in tags:
        for link in released_links(tag):
            if is_own(link.url):
                released[link.url].append(link)
    released_broken = [
        f"{url}: {reason} ({', '.join(f'{link.source}:{link.line}' for link in links)})"
        for url, links in sorted(released.items())
        if (reason := checker.problem(url))
    ]
    shown = (
        build.relative_to(ROOT) if build.is_relative_to(ROOT) else build
    ).as_posix()
    span = f"{tags[-1]} to {tags[0]}" if len(tags) > 1 else tags[0]
    findings = {
        f"Links that lead nowhere in the build of the documentation ({shown}) or "
        "the working tree": broken,
        f"Links of the releases ({span}), on PyPI and in the installed copies, "
        "that lead nowhere in the build or the working tree": released_broken,
    }
    for title, lines in findings.items():
        if lines:
            print(title)
            for line in lines:
                print(f"  {line}")
            print()
    if broken or released_broken:
        print(
            "Fix the link, or the page, anchor or file that it names. A link of a "
            "release cannot be changed: keep what it leads to."
        )
        return 1
    print(
        f"Each of the {counts['own']} links to the documentation site and the "
        f"repository leads somewhere in the build ({shown}) and the working tree, "
        f"as do the {counts['relative']} relative links of the Markdown files and "
        f"the {len(released)} links of the releases ({span})."
    )
    return 0


def is_checked_externally(url: str) -> bool:
    """
    Return whether the external check asks for a link: one that the offline
    check does not take (``is_own()``), to a host that can be public.

    :param url: The link.
    :type url: str
    :returns: Whether to ask for it.
    :rtype: bool
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    return not is_own(url) and not _LOCAL_HOST.fullmatch(host)


class _NotPublicError(Exception):
    """
    A request or a redirect to a host that is not public, or to another scheme
    than http and https, which the check does not send.
    """


def resolves_publicly(host: str) -> bool:
    """
    Return whether each address that a host resolves to is public (global): not
    one of a local network, of this machine (loopback), link-local, e.g. a
    cloud's metadata service at 169.254.169.254, or reserved. A host that does
    not resolve counts as public, for the request to report it.

    :param host: The name or address of the host.
    :type host: str
    :returns: Whether the host is public.
    :rtype: bool
    """
    try:
        found = socket.getaddrinfo(host, None)
    except OSError:
        return True
    addresses = {str(info[4][0]).split("%", 1)[0] for info in found}
    return all(ipaddress.ip_address(address).is_global for address in addresses)


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    """
    Follows a redirect only to an http or https address of a public host:
    urllib's own handler follows one to any host, and to ftp.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """
        Return the request to the address that a site redirects to.

        :param req: The request that was redirected.
        :type req: urllib.request.Request
        :param fp: The answer.
        :type fp: IO[bytes]
        :param code: The status of the answer.
        :type code: int
        :param msg: The reason phrase of the answer.
        :type msg: str
        :param headers: The headers of the answer.
        :type headers: HTTPMessage
        :param newurl: The address redirected to, joined with that of the
            request.
        :type newurl: str
        :returns: The new request.
        :rtype: urllib.request.Request | None
        :raises _NotPublicError: If the address is not an http or https one of a
            public host.
        """
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme not in ("http", "https"):
            raise _NotPublicError(f"not asked: redirected to {parts.scheme}")
        if not resolves_publicly(parts.hostname or ""):
            raise _NotPublicError("not asked: redirected to a host that is not public")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener() -> urllib.request.OpenerDirector:
    """
    Return what sends the requests: urllib's handlers of http and https only,
    without those of files, ftp and data URLs, with the redirects of
    ``_PublicRedirects`` and the certificates verified. It keeps no cookies and
    sends no credentials.

    :returns: The opener.
    :rtype: urllib.request.OpenerDirector
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.UnknownHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        urllib.request.HTTPDefaultErrorHandler(),
        _PublicRedirects(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


_OPENER = _opener()


def fetch(url: str) -> tuple[str, str]:
    """
    Ask a site for a page, as a browser asks, again after a while when it limits
    the rate, fails or does not answer.

    :param url: The link, without its anchor.
    :type url: str
    :returns: The verdict, ``"ok"``, ``"broken"`` (404, 410, or a host that does
        not exist) or ``"unchecked"`` (another error, or a host that is not
        public, which is not asked), and what the site answered, e.g.
        ``HTTP 404``.
    :rtype: tuple[str, str]
    """
    if not resolves_publicly(urllib.parse.urlsplit(url).hostname or ""):
        return "unchecked", "not asked: the host is not public"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en",
        },
    )
    answer = ""
    for attempt in range(_ATTEMPTS):
        if attempt:
            time.sleep(_retry_delay(answer, attempt))
        try:
            with _OPENER.open(request, timeout=_TIMEOUT):
                return "ok", ""
        except _NotPublicError as e:
            return "unchecked", str(e)
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return "broken", f"HTTP {e.code}"
            answer = f"HTTP {e.code}"
            retry_after = e.headers.get("Retry-After", "") if e.headers else ""
            if e.code != 429 and e.code < 500:
                return "unchecked", answer
            if retry_after.isdigit():
                answer += f", retry after {retry_after} s"
        except urllib.error.URLError as e:
            if _host_not_found(e.reason):
                return "broken", "host not found"
            answer = str(e.reason)
        except (TimeoutError, OSError) as e:
            answer = str(e) or type(e).__name__
    return "unchecked", answer


def fetch_all(
    urls: list[str], deadline: float = _DEADLINE
) -> dict[str, tuple[str, str]]:
    """
    Ask for the pages, ``_WORKERS`` at a time, and return the verdict of each
    (``fetch()``). A page without an answer within the deadline is not checked.
    The requests run in daemon threads, which the process leaves behind at its
    exit, so that a request that does not end cannot hold it.

    :param urls: The links, without their anchors.
    :type urls: list[str]
    :param deadline: The longest wait for all the answers, in seconds.
    :type deadline: float
    :returns: The verdict of each link.
    :rtype: dict[str, tuple[str, str]]
    """
    pending: queue.SimpleQueue[str] = queue.SimpleQueue()
    for url in urls:
        pending.put(url)
    verdicts: dict[str, tuple[str, str]] = {}

    def work() -> None:
        """
        Ask for the pages of the queue until it is empty.
        """
        while True:
            try:
                url = pending.get_nowait()
            except queue.Empty:
                return
            verdicts[url] = fetch(url)

    workers = [threading.Thread(target=work, daemon=True) for _ in range(_WORKERS)]
    for worker in workers:
        worker.start()
    end = time.monotonic() + deadline
    for worker in workers:
        worker.join(max(0.0, end - time.monotonic()))
    answered = dict(verdicts)
    late = ("unchecked", f"no answer within {deadline:g} s")
    return {url: answered.get(url, late) for url in urls}


def _retry_delay(answer: str, attempt: int) -> float:
    """
    Return how long to wait before asking again: what the site's Retry-After
    asked for, up to ``_MAX_RETRY_AFTER``, else 5 s, then 15 s.

    :param answer: The last answer, e.g. ``HTTP 429, retry after 20 s``.
    :type answer: str
    :param attempt: The number of the next attempt, from 1.
    :type attempt: int
    :returns: The delay, in seconds.
    :rtype: float
    """
    if found := re.search(r"retry after (\d+) s", answer):
        return min(int(found.group(1)), _MAX_RETRY_AFTER)
    return 5.0 if attempt == 1 else 15.0


def _host_not_found(reason: object) -> bool:
    """
    Return whether a failed request found no host by the name of the link.

    :param reason: The reason of the ``URLError``.
    :type reason: object
    :returns: Whether the name is not found.
    :rtype: bool
    """
    return isinstance(reason, socket.gaierror) and reason.errno == getattr(
        socket, "EAI_NONAME", None
    )


def _online() -> bool:
    """
    Return whether this machine finds hosts by name, e.g. github.com: offline,
    each link would be a host that does not exist.

    :returns: Whether it is online.
    :rtype: bool
    """
    try:
        socket.getaddrinfo("github.com", 443)
    except OSError:
        return False
    return True


def check_external() -> int:
    """
    Ask the other sites for the pages that the files link to, and print what
    leads nowhere, and what could not be checked.

    :returns: The exit code: 0 if no link leads nowhere, else 1.
    :rtype: int
    :raises _CheckError: If this machine is offline.
    """
    if not _online():
        raise _CheckError("Cannot find github.com: the links need the network.")
    places: dict[str, list[Link]] = collections.defaultdict(list)
    ignored_used: set[str] = set()
    for path in _source_files(ROOT):
        name = path.relative_to(ROOT).as_posix()
        for link in find_urls(name, path.read_text(encoding="utf-8")):
            if not is_checked_externally(link.url):
                continue
            ignoring = [start for start in IGNORED if link.url.startswith(start)]
            if ignoring:
                ignored_used.update(ignoring)
                continue
            places[link.url.split("#", 1)[0]].append(link)
    verdicts = fetch_all(list(places))
    broken = []
    unchecked = []
    for url, (verdict, answer) in sorted(verdicts.items()):
        where = ", ".join(f"{link.source}:{link.line}" for link in places[url])
        if verdict == "broken":
            broken.append(f"{url}: {answer} ({where})")
        elif verdict == "unchecked":
            unchecked.append(f"{url}: {answer} ({where})")
    findings = {
        "Links to other sites that lead nowhere": broken,
        "IGNORED: entries that match no link, to remove": sorted(
            IGNORED.keys() - ignored_used
        ),
    }
    for title, lines in findings.items():
        if lines:
            print(title)
            for line in lines:
                print(f"  {line}")
            print()
    if unchecked:
        print(
            "Warning: links not checked, as the site refused the request, limited "
            "the rate or did not answer in time, or as the host is not public: open "
            "them in a browser"
        )
        for line in unchecked:
            print(f"  {line}")
        print()
    if any(findings.values()):
        print(
            "Fix each link: the page may have moved (search its site), or leave it "
            "out. A link that leads nowhere on purpose, e.g. a placeholder, goes in "
            "IGNORED with the reason."
        )
        return 1
    print(
        f"None of the {len(verdicts)} links to other sites leads nowhere"
        + (f", but {len(unchecked)} could not be checked." if unchecked else ".")
    )
    return 0


def main(arguments: Iterable[str] | None = None) -> int:
    """
    Run the check that the arguments ask for.

    :param arguments: The command-line arguments, by default those of the
        process.
    :type arguments: Iterable[str] | None
    :returns: The exit code.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--build",
        type=Path,
        default=DEFAULT_BUILD,
        help="the build of the documentation (dirhtml), by default docs/build/dirhtml",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="check the links to other sites instead, online",
    )
    options = parser.parse_args(None if arguments is None else list(arguments))
    try:
        if options.external:
            return check_external()
        return check_own(options.build.resolve())
    except _CheckError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
