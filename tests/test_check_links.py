# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the check of the links (dev/check_links.py).
"""

from __future__ import annotations

import email.message
import http.client
import io
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dev import check_links
from dev.check_links import (
    OLD_SITE,
    REPOSITORY,
    SITE,
    Link,
    OwnLinks,
    fetch,
    fetch_all,
    find_relative_links,
    find_urls,
    github_anchor,
    is_checked_externally,
    is_own,
    main,
    markdown_anchors,
    resolves_publicly,
)


# The variation selector after an emoji such as ⚙ or ℹ, which GitHub keeps in
# the anchor
_VS16 = chr(0xFE0F)

# Headings of a Markdown file, and the anchors that GitHub's Markdown API gave
# them (2026-10-22)
_GITHUB_ANCHORS = [
    (f"{chr(0x2699)}{_VS16} Configuration", f"{_VS16}-configuration"),
    (f"{chr(0x1F4E6)} Installation", "-installation"),
    (f"{chr(0x1F9EA)} VCR Cassette Management", "-vcr-cassette-management"),
    ("[1.1.0] - 2026-10-17", "110---2026-10-17"),
    ("Foo_bar `code` *em* **strong**", "foo_bar-code-em-strong"),
    ("Ünïcödé — dash & more", "ünïcödé--dash--more"),
    ("A [link](https://x.y) here", "a-link-here"),
    ("Tab\there", "tabhere"),
    (f"{chr(0x2139)}{_VS16} Info", f"{chr(0x2139)}{_VS16}-info"),
    ("C++ / C#: 3.14?", "c--c-314"),
]


@pytest.fixture
def own(tmp_path: Path) -> OwnLinks:
    """
    Returns the check of the own links against a small build of the
    documentation and a small working tree.

    :param tmp_path: A temporary folder.
    :type tmp_path: Path
    :returns: The check, which knows the tags v1.0.0 and v1.1.0.
    :rtype: OwnLinks
    """
    build = tmp_path / "build"
    page = build / "documents" / "docling"
    page.mkdir(parents=True)
    (build / "index.html").write_text('<h1 id="top">ContextGem</h1>', "utf-8")
    (page / "index.html").write_text(
        '<section id="supported-formats"><h2>Formats</h2></section>', "utf-8"
    )
    (build / "_static").mkdir()
    (build / "_static" / "figure.svg").write_text("<svg/>", "utf-8")
    root = tmp_path / "root"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "docs.yml").write_text("", "utf-8")
    (root / "docs").mkdir()
    (root / "CONTRIBUTING.md").write_text(
        f"# Contributing\n\n## {chr(0x1F9EA)} VCR Cassettes\n\n"
        '```\n# A comment, not a heading\n```\n\n<a id="custom"></a>\n',
        "utf-8",
    )
    (root / "fabfile.py").write_text("a\nb\nc\n", "utf-8")
    (tmp_path / "outside.md").write_text("# Outside\n", "utf-8")
    return OwnLinks(build, root, {"v1.0.0", "v1.1.0", "HEAD"})


class _Answers:
    """
    Stands in for the opener of the check: answers each request with the next of
    its answers, and records the requests.
    """

    def __init__(self, *answers: BaseException | None) -> None:
        """
        :param answers: An exception to raise, or ``None`` for a page.
        :type answers: BaseException | None
        """
        self.answers = list(answers)
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        """
        Answers a request.

        :param request: The request.
        :type request: urllib.request.Request
        :param timeout: The timeout of the request.
        :type timeout: float
        :returns: An empty page.
        :rtype: io.BytesIO
        """
        self.requests.append(request)
        answer = self.answers.pop(0)
        if answer is not None:
            raise answer
        return io.BytesIO(b"")


def _http_error(code: int, retry_after: str = "") -> urllib.error.HTTPError:
    """
    Returns the error that ``urlopen`` raises on an HTTP status.

    :param code: The status.
    :type code: int
    :param retry_after: The Retry-After header, if any.
    :type retry_after: str
    :returns: The error.
    :rtype: urllib.error.HTTPError
    """
    headers = email.message.Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://a.org/", code, "", headers, None)


class TestCheckLinks:
    """
    Tests for the check of the links.
    """

    @pytest.mark.parametrize(("heading", "anchor"), _GITHUB_ANCHORS)
    def test_github_anchor_is_the_anchor_of_github(self, heading: str, anchor: str):
        """
        Tests that the anchor of a heading is the one that GitHub gives it: the
        letters, marks (an emoji's variation selector), numbers, underscores and
        hyphens kept, and each space a hyphen.
        """
        assert github_anchor(heading) == anchor

    def test_markdown_anchors_number_repeated_headings(self):
        """
        Tests that a repeated heading gets a number after its anchor, that the
        ids of HTML elements are anchors, and that a line of a code block that
        starts with ``#`` is no heading.
        """
        text = (
            "# Title\n\n## Usage\n\n## Usage ##\n\n```bash\n# install\n```\n\n"
            '<a id="custom"></a> <a name="named"></a>\n\n~~~\n## Fenced\n~~~\n'
        )
        assert markdown_anchors(text) == {
            "title",
            "usage",
            "usage-1",
            "custom",
            "named",
        }

    def test_find_urls_leaves_out_placeholders_and_inline_code(self):
        """
        Tests that the links of a text are found up to their end, without the
        punctuation after them, and that an address with a placeholder, or in
        inline code of a Markdown or reST file, is left out, but not one in the
        code of a Python file. In reST, single backticks make a link, not code.
        """
        markdown = (
            "See https://a.org/x. Or (https://b.org/y), **https://c.org/z**\n"
            "`https://e.org/code` [d](https://d.org/p)\n"
            "https://{resource}.azure.com and https://g.org/accounts/<id>/run\n"
        )
        assert list(find_urls("README.md", markdown)) == [
            Link("https://a.org/x", "README.md", 1),
            Link("https://b.org/y", "README.md", 1),
            Link("https://c.org/z", "README.md", 1),
            Link("https://d.org/p", "README.md", 2),
        ]
        assert [link.url for link in find_urls("module.py", markdown)] == [
            "https://a.org/x",
            "https://b.org/y",
            "https://c.org/z",
            "https://e.org/code",
            "https://d.org/p",
        ]
        rest = "`Docs <https://d.org/p>`_ and ``https://f.org/literal``\n"
        assert list(find_urls("page.rst", rest)) == [
            Link("https://d.org/p", "page.rst", 1)
        ]

    def test_find_relative_links_of_markdown(self):
        """
        Tests that the relative links of a Markdown file are found in its links,
        its link definitions and the attributes of its HTML, but not in its code,
        and that absolute links and other schemes are left to the other checks.
        """
        text = (
            "[a](CONTRIBUTING.md#x) [b](https://a.org) [c](mailto:a@b.org)\n"
            "[d]: docs/figure.svg\n"
            '<img src="assets/mark.svg"> <a href="//cdn.org/x">e</a>\n'
            "`[f](in_code.md)` [g](#local)\n"
            "```\n[h](fenced.md)\n```\n"
        )
        assert list(find_relative_links("README.md", text)) == [
            Link("CONTRIBUTING.md#x", "README.md", 1),
            Link("docs/figure.svg", "README.md", 2),
            Link("assets/mark.svg", "README.md", 3),
            Link("#local", "README.md", 4),
        ]

    def test_site_links_lead_to_the_pages_and_anchors_of_the_build(self, own: OwnLinks):
        """
        Tests that a link to the documentation site, or to its old domain, leads
        to a page or a file of the build, and its anchor to an element of the
        page.
        """
        for url in (
            SITE,
            f"{SITE}#top",
            f"{SITE}?highlight=x",
            f"{SITE}documents/docling/",
            f"{SITE}documents/docling",
            f"{SITE}documents/docling/#supported-formats",
            f"{OLD_SITE}documents/docling/#supported-formats",
            f"{SITE}_static/figure.svg",
        ):
            assert is_own(url)
            assert own.problem(url) is None, url
        assert own.problem(f"{SITE}documents/docling/#formats") == (
            "no anchor #formats on the page"
        )
        assert own.problem(f"{SITE}documents/pdf/") == (
            "no such page in the build of the documentation"
        )
        # The file exists, next to the build: a path that leads out of it is not
        # followed, as it would read the other files of the machine
        assert own.problem(f"{SITE}../outside.md") == (
            "leads out of the build of the documentation"
        )
        assert own.problem(f"{SITE}%2E%2E/outside.md") == (
            "leads out of the build of the documentation"
        )

    def test_repository_links_lead_to_files_tags_and_workflows(self, own: OwnLinks):
        """
        Tests that a link to a file of the repository on main leads to a file of
        the working tree, and its anchor to a heading, an element or a line; that
        a compare or release link names known tags; and that a link to a
        workflow leads to its file.
        """
        raw = "https://raw.githubusercontent.com/shcherbak-ai/contextgem/"
        for url in (
            f"{REPOSITORY}/blob/main/CONTRIBUTING.md",
            f"{REPOSITORY}/blob/main/CONTRIBUTING.md#-vcr-cassettes",
            f"{REPOSITORY}/blob/main/CONTRIBUTING.md#custom",
            f"{REPOSITORY}/blob/main/fabfile.py#L2-L3",
            f"{REPOSITORY}/tree/main/docs",
            f"{raw}main/fabfile.py",
            f"{raw}refs/heads/main/fabfile.py",
            f"{REPOSITORY}/compare/v1.0.0...v1.1.0",
            f"{REPOSITORY}/compare/v1.1.0...HEAD",
            f"{REPOSITORY}/releases/tag/v1.1.0",
            f"{REPOSITORY}/actions/workflows/docs.yml",
            f"{REPOSITORY}/actions/workflows/docs.yml/badge.svg?branch=main",
        ):
            assert is_own(url)
            assert own.problem(url) is None, url
        problems = {
            f"{REPOSITORY}/blob/main/CONTRIBUTING.md#a-comment-not-a-heading": (
                "no heading or element with the anchor #a-comment-not-a-heading"
            ),
            f"{REPOSITORY}/blob/main/fabfile.py#L4": "no line 4 in the file",
            f"{REPOSITORY}/blob/main/README.md": "no such file in the working tree",
            f"{REPOSITORY}/blob/main/docs": "no such file in the working tree",
            f"{raw}dev/fabfile.py": "not a link to main",
            f"{REPOSITORY}/compare/v1.1.0...v1.2.0": "no tag v1.2.0",
            f"{REPOSITORY}/releases/tag/v2.0.0": "no such tag",
            f"{REPOSITORY}/actions/workflows/ci.yml": "no such workflow",
            f"{REPOSITORY}/blob/main/../outside.md": "leads out of the repository",
        }
        for url, problem in problems.items():
            assert own.problem(url) == problem, url

    def test_other_links_of_the_repository_are_external(self):
        """
        Tests that the links to the repository that name no file, workflow,
        comparison or release, e.g. its issues, are left to the external check,
        as are the links to another repository or site whose address starts
        like those of ContextGem.
        """
        for url in (
            REPOSITORY,
            f"{REPOSITORY}.git",
            f"{REPOSITORY}#readme",
            f"{REPOSITORY}/issues",
            f"{REPOSITORY}/blob/dev/README.md",
            "https://github.com/shcherbak-ai/contextgem-other/blob/main/README.md",
            "https://contextgem.shcherbak.ai.example.org/",
        ):
            assert not is_own(url), url
        assert is_checked_externally(f"{REPOSITORY}/issues")

    def test_relative_links_lead_to_files_of_the_repository(self, own: OwnLinks):
        """
        Tests that a relative link of a Markdown file leads to a file of the
        repository, and its anchor to a heading of the file, or of the same file
        for an anchor alone.
        """
        source = own.root / "CONTRIBUTING.md"
        problems = {
            "#-vcr-cassettes": None,
            "#vcr": "no heading or element with the anchor #vcr",
            "fabfile.py#L3": None,
            "docs": None,
            "/docs": None,
            ".github/workflows/docs.yml": None,
            "README.md": "no such file in the working tree",
            "../outside.md": "leads out of the repository",
        }
        for target, problem in problems.items():
            link = Link(target, "CONTRIBUTING.md", 1)
            assert own.relative_problem(link, source) == problem, target

    def test_hosts_that_cannot_be_public_are_not_asked(self):
        """
        Tests that the external check leaves out local addresses and the names
        reserved for examples and tests, and the links that the offline check
        takes.
        """
        for url in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:9120/",
            "https://example.com/page",
            "https://api.example.org/",
            "https://dummy-endpoint.local",
            "https://server.test/",
            f"{SITE}documents/docling/",
        ):
            assert not is_checked_externally(url), url
        assert is_checked_externally("https://github.com/docling-project/docling")

    @pytest.mark.parametrize(
        ("answers", "verdict", "attempts"),
        [
            ([None], ("ok", ""), 1),
            ([_http_error(404)], ("broken", "HTTP 404"), 1),
            ([_http_error(410)], ("broken", "HTTP 410"), 1),
            ([_http_error(403)], ("unchecked", "HTTP 403"), 1),
            ([_http_error(503), None], ("ok", ""), 2),
            ([_http_error(503)] * 3, ("unchecked", "HTTP 503"), 3),
            ([TimeoutError()] * 3, ("unchecked", "TimeoutError"), 3),
            (
                [urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "unknown"))],
                ("broken", "host not found"),
                1,
            ),
        ],
    )
    def test_fetch_fails_only_a_page_or_host_that_is_not_there(
        self,
        monkeypatch: pytest.MonkeyPatch,
        answers: list[BaseException | None],
        verdict: tuple[str, str],
        attempts: int,
    ):
        """
        Tests that a link fails on 404, 410 or a host that does not exist; that a
        site that refuses the request is not asked again; and that a failing
        site or a timeout is asked again, and reported as not checked in the
        end. The request is sent as a browser sends it.
        """
        stub = _Answers(*answers)
        monkeypatch.setattr(check_links, "_OPENER", stub)
        monkeypatch.setattr(check_links, "resolves_publicly", lambda host: True)
        monkeypatch.setattr(check_links.time, "sleep", lambda seconds: None)
        assert fetch("https://a.org/page") == verdict
        assert len(stub.requests) == attempts
        assert "Mozilla/5.0" in (stub.requests[0].get_header("User-agent") or "")

    def test_fetch_waits_as_long_as_the_site_asks_within_a_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Tests that a site that limits the rate is asked again after the time of
        its Retry-After, waiting no longer than the limit.
        """
        waits: list[float] = []
        stub = _Answers(_http_error(429, "2"), _http_error(429, "600"), None)
        monkeypatch.setattr(check_links, "_OPENER", stub)
        monkeypatch.setattr(check_links, "resolves_publicly", lambda host: True)
        monkeypatch.setattr(check_links.time, "sleep", waits.append)
        assert fetch("https://a.org/page") == ("ok", "")
        assert waits == [2, check_links._MAX_RETRY_AFTER]

    def test_main_fails_without_the_build(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        """
        Tests that the check of the own links fails, and says what to do, when
        the folder of the build has no documentation.
        """
        assert main(["--build", str(tmp_path)]) == 1
        assert "build it first (uv run fab docs)" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("addresses", "public"),
        [
            (["140.82.112.3"], True),
            (["2606:50c0:8000::153"], True),
            (["192.168.1.1"], False),
            (["10.0.0.5"], False),
            (["127.0.0.1"], False),
            (["169.254.169.254"], False),
            (["::1"], False),
            (["fd00::1"], False),
            (["fe80::1%eth0"], False),
            (["140.82.112.3", "10.0.0.5"], False),
        ],
    )
    def test_resolves_publicly_refuses_hosts_of_a_local_network(
        self, monkeypatch: pytest.MonkeyPatch, addresses: list[str], public: bool
    ):
        """
        Tests that a host is public only when each address it resolves to is: not
        one of a local network, of the machine, or link-local, e.g. a cloud's
        metadata service.
        """
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port: [(0, 0, 0, "", (address, 0)) for address in addresses],
        )
        assert resolves_publicly("host.org") is public

    def test_resolves_publicly_leaves_an_unknown_host_to_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Tests that a host that does not resolve counts as public, so that the
        request reports it as a host that does not exist.
        """

        def unknown(host: str, port: None) -> list:
            """
            Fails to resolve a host.
            """
            raise socket.gaierror(socket.EAI_NONAME, "unknown")

        monkeypatch.setattr(socket, "getaddrinfo", unknown)
        assert resolves_publicly("no-such-host.org")

    def test_fetch_does_not_ask_a_host_that_is_not_public(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Tests that a link to a host that is not public is reported as not
        checked, without a request.
        """
        stub = _Answers()
        monkeypatch.setattr(check_links, "_OPENER", stub)
        monkeypatch.setattr(check_links, "resolves_publicly", lambda host: False)
        assert fetch("http://router.lan/reboot") == (
            "unchecked",
            "not asked: the host is not public",
        )
        assert stub.requests == []

    def test_redirects_lead_only_to_public_http_hosts(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Tests that the check follows a redirect to an http or https address of a
        public host, and refuses one to a host that is not public, or to another
        scheme, which urllib's own handler follows (ftp).
        """
        monkeypatch.setattr(
            check_links, "resolves_publicly", lambda host: host == "public.org"
        )
        handler = check_links._PublicRedirects()
        request = urllib.request.Request("https://a.org/page")
        headers = http.client.HTTPMessage()

        def redirect(url: str) -> urllib.request.Request | None:
            """
            Asks the handler to follow a redirect to an address.
            """
            return handler.redirect_request(
                request, io.BytesIO(), 302, "Found", headers, url
            )

        followed = redirect("https://public.org/new")
        assert followed is not None
        assert followed.full_url == "https://public.org/new"
        with pytest.raises(check_links._NotPublicError, match="not public"):
            redirect("http://169.254.169.254/latest/meta-data/")
        with pytest.raises(check_links._NotPublicError, match="redirected to ftp"):
            redirect("ftp://public.org/file")

    def test_a_refused_redirect_is_reported_as_not_checked(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Tests that a link whose site redirects to a host that is not public is
        reported as not checked, and is not asked again.
        """
        stub = _Answers(check_links._NotPublicError("not asked: redirected to a host"))
        monkeypatch.setattr(check_links, "_OPENER", stub)
        monkeypatch.setattr(check_links, "resolves_publicly", lambda host: True)
        assert fetch("https://a.org/page") == (
            "unchecked",
            "not asked: redirected to a host",
        )
        assert len(stub.requests) == 1

    def test_fetch_all_does_not_wait_past_the_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Tests that the links are asked for together, and that a link without an
        answer within the deadline, e.g. of a site that hangs, is reported as not
        checked, without holding the check.
        """
        release = threading.Event()

        def answer(url: str) -> tuple[str, str]:
            """
            Answers at once, but for the site that hangs, which answers once
            released.
            """
            if "hangs" in url:
                release.wait(30)
            return "ok", ""

        monkeypatch.setattr(check_links, "fetch", answer)
        urls = ["https://hangs.org/", *(f"https://site{i}.org/" for i in range(20))]
        started = time.monotonic()
        verdicts = fetch_all(urls, deadline=0.5)
        release.set()
        assert time.monotonic() - started < 10
        assert verdicts["https://hangs.org/"] == (
            "unchecked",
            "no answer within 0.5 s",
        )
        assert [verdicts[url] for url in urls[1:]] == [("ok", "")] * 20

    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "ftp://files.org/file", "data:,secret"]
    )
    def test_the_opener_speaks_http_and_https_only(self, url: str):
        """
        Tests that what sends the requests refuses another scheme than http and
        https, e.g. a file of the machine, without a request.
        """
        with pytest.raises(urllib.error.URLError, match="unknown url type"):
            check_links._OPENER.open(url, timeout=1)
