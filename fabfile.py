# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Fabric automation tasks for ContextGem development.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import io
import json
import os
import posixpath
import re
import shutil
import socket
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from html import escape
from http import HTTPStatus
from pathlib import Path

from fabric import task
from invoke.context import Context
from invoke.exceptions import Exit


_ROOT = Path(__file__).parent

_README = _ROOT / "README.md"

# The repository's social preview: the drawing, and the picture made from it.
_SOCIAL_PREVIEW = _ROOT / "social" / "github-social-preview.svg"

# The browsers that can take that picture, by the names they have on a PATH
# and by where Windows and macOS install them.
_BROWSER_NAMES = (
    "chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "msedge",
    "microsoft-edge",
)
_BROWSER_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)


def _find_browser() -> str | None:
    """Return a Chrome, Edge or Chromium on this machine, or ``None``."""
    for name in _BROWSER_NAMES:
        if found := shutil.which(name):
            return found
    return next((path for path in _BROWSER_PATHS if Path(path).is_file()), None)


def _port_in_use(port: int) -> bool:
    """Return whether something already answers on ``localhost:port``."""
    for family, kind, proto, _, address in socket.getaddrinfo(
        "localhost", port, type=socket.SOCK_STREAM
    ):
        with socket.socket(family, kind, proto) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(address) == 0:
                return True
    return False


# GitHub's Markdown API renders the README's HTML, in the mode of a file in a
# repository. Without a token it answers 60 requests an hour.
_MARKDOWN_API = "https://api.github.com/markdown"

# github-markdown-css gives the page GitHub's look, one stylesheet per theme;
# npm installs it (package.json).
_MARKDOWN_CSS = "node_modules/github-markdown-css/github-markdown-{scheme}.css"

# The preview's two pages, each in one of GitHub's themes whatever the browser
# is set to, so that both can be open side by side, with GitHub's ground in it.
# The light one is served on the task's port, the dark one on the next.
_PREVIEW_SCHEMES = {"light": "#ffffff", "dark": "#0d1117"}

# What the preview serves besides its page: the stylesheets, and the folders the
# README's images come from. Nothing else in the repository is reachable.
_PREVIEW_PATHS = (
    "node_modules/github-markdown-css/",
    "docs/source/_static/",
    "assets/",
)

# The page around GitHub's HTML. The title names the project and the theme, so
# that an agent can tell this preview from another project's. The column is as
# wide as the README's on the repository's page on GitHub at 1280 pixels and
# wider, and below 768, where the sidebar goes under it; in between, GitHub's is
# narrower. Links are underlined, as GitHub has them.
_PREVIEW_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="{scheme}">
<title>README.md · contextgem · {scheme}</title>
<link rel="stylesheet" href="/{stylesheet}">
<style>
  body {{ margin: 0; background: {background}; }}
  .markdown-body {{
    box-sizing: content-box; max-width: 838px; margin: 0 auto; padding: 32px 33px;
  }}
  .markdown-body a {{ text-decoration: underline; }}
</style>
</head>
<body>
<article class="markdown-body">
<!-- README -->
</article>
</body>
</html>
"""


class _RenderError(Exception):
    """GitHub did not render the README; the message says why."""


# An alert (``> [!NOTE]``) as the API's "markdown" mode returns it, a quote, and
# as its "gfm" mode does, which is how the repository's page shows it.
_ALERT_QUOTE = re.compile(
    r"<blockquote>\s*<p>\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]"
)
_ALERT_DIV = re.compile(r'<div class="markdown-alert ')


@functools.lru_cache(maxsize=1)
def _render_readme(text: str) -> str:
    """
    Return the HTML GitHub makes of ``text`` as a repository's README.

    The "markdown" mode gives the headings their anchors, and the "gfm" mode
    gives the alerts, which the other returns as quotes: where there are alerts,
    they are taken from a second request. The last result is kept, so a page
    reload asks GitHub again only when the README has changed.
    """
    html = _render_markdown(text, "markdown")
    quotes = _element_spans(html, _ALERT_QUOTE, "blockquote")
    if quotes:
        gfm = _render_markdown(text, "gfm")
        alerts = [
            gfm[start:end] for start, end in _element_spans(gfm, _ALERT_DIV, "div")
        ]
        if len(alerts) == len(quotes):
            for (start, end), alert in reversed(list(zip(quotes, alerts, strict=True))):
                html = html[:start] + alert + html[end:]
    return html


def _element_spans(
    html: str, opening: re.Pattern[str], tag: str
) -> list[tuple[int, int]]:
    """Return where each element that ``opening`` starts begins and ends, nested ones included."""
    spans = []
    tags = re.compile(rf"<(/?){tag}\b[^>]*>")
    for match in opening.finditer(html):
        depth = 0
        for found in tags.finditer(html, match.start()):
            depth += -1 if found.group(1) else 1
            if depth == 0:
                spans.append((match.start(), found.end()))
                break
    return spans


def _render_markdown(text: str, mode: str) -> str:
    """Return the HTML that GitHub's Markdown API makes of ``text`` in ``mode``."""
    request = urllib.request.Request(
        _MARKDOWN_API,
        data=json.dumps({"text": text, "mode": mode}).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.headers.get("x-ratelimit-remaining") == "0":
            reset = datetime.fromtimestamp(
                int(error.headers.get("x-ratelimit-reset", "0"))
            )
            raise _RenderError(
                f"GitHub's limit of 60 requests an hour is used up until {reset:%H:%M}."
            ) from error
        body = error.read().decode("utf-8", "replace")
        raise _RenderError(
            f"GitHub's Markdown API answered {error.code}: {body}"
        ) from error
    except OSError as error:
        raise _RenderError(
            f"GitHub's Markdown API could not be reached: {error}"
        ) from error


class _ReadmePreview(http.server.SimpleHTTPRequestHandler):
    """Serves README.md as GitHub renders it at ``/``, and the files its page loads."""

    def __init__(self, *args, scheme: str, **kwargs) -> None:
        # Set before the base class's __init__, which handles the request.
        self.scheme = scheme
        super().__init__(*args, **kwargs)

    # Windows can map these types to others in its registry, and a browser
    # refuses a stylesheet that is not served as text/css.
    extensions_map = {  # noqa: RUF012 - overrides the attribute of the handler
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".css": "text/css",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }

    def send_head(self):
        """Answer ``/`` with the page, a path under ``_PREVIEW_PATHS`` with its file."""
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        path = posixpath.normpath(path).lstrip("/")
        if not path:
            return self._send_page()
        if path.startswith(_PREVIEW_PATHS):
            return super().send_head()
        self.send_error(
            HTTPStatus.NOT_FOUND, "The preview serves the README and its images only"
        )
        return None

    def end_headers(self) -> None:
        """Ask the browser not to keep anything, so a reload shows an edited image."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_page(self) -> io.BytesIO:
        """Send the headers of the page, and return its body."""
        try:
            html = _render_readme(_README.read_text(encoding="utf-8"))
            frame = _PREVIEW_PAGE.format(
                scheme=self.scheme,
                stylesheet=_MARKDOWN_CSS.format(scheme=self.scheme),
                background=_PREVIEW_SCHEMES[self.scheme],
            )
            status, page = HTTPStatus.OK, frame.replace("<!-- README -->", html)
        except _RenderError as error:
            print(error)
            status, page = HTTPStatus.BAD_GATEWAY, f"<pre>{escape(str(error))}</pre>"
        body = page.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return io.BytesIO(body)


# =============================================================================
# Development Setup Tasks
# =============================================================================


@task
def sync(c: Context) -> None:
    """Upgrade dependencies within pyproject.toml constraints (rewrites uv.lock)."""
    c.run("uv sync --all-groups --upgrade")


@task
def setup(c: Context) -> None:
    """Set up development environment (locked deps, npm tools, hooks)."""
    print("Installing dependencies...")
    # The versions locked in uv.lock, i.e. the ones that CI uses: no upgrades
    c.run("uv sync --all-groups")
    print("\nInstalling the development tools of package.json...")
    # The versions locked in package-lock.json: the stylesheet of the README
    # preview and the agents' browser
    c.run("npm ci")
    print("\nInstalling pre-commit hooks...")
    install_hooks(c)
    print("\n✓ Development environment ready!")


# =============================================================================
# Pre-commit Tasks
# =============================================================================


@task
def install_hooks(c: Context) -> None:
    """Install pre-commit hooks (run once after cloning)."""
    c.run("uv run pre-commit install")
    c.run("uv run pre-commit install --hook-type commit-msg")
    print("\n✓ Pre-commit hooks installed successfully!")


@task
def lint(c: Context) -> None:
    """Run pre-commit checks on all files."""
    c.run("uv run pre-commit run --all-files")


# =============================================================================
# Documentation Tasks
# =============================================================================


def _clear_docs_build(docs_dir: str) -> None:
    """
    Remove the previous build, so that no file of a removed page or of an earlier
    configuration is served with the new one.
    """
    shutil.rmtree(os.path.join(docs_dir, "build", "dirhtml"), ignore_errors=True)


@task
def docs(c: Context) -> None:
    """Build documentation (static build with full rebuild)."""
    docs_dir = os.path.join(os.getcwd(), "docs")
    _clear_docs_build(docs_dir)
    cmd = "uv run sphinx-build -b dirhtml source build/dirhtml -v -E -W"
    with c.cd(docs_dir):
        c.run(cmd)
    print("\n✓ Documentation built successfully!")
    print("  To view, serve the folder and open http://127.0.0.1:9000/:")
    print(
        "  uv run python -m http.server 9000 --bind 127.0.0.1"
        " --directory docs/build/dirhtml"
    )


@task(name="docs-live")
def docs_live(c: Context, port: int = 9000) -> None:
    """
    Start live documentation server with auto-reload.

    Args:
        port: Port number for the dev server (default: 9000)

    Example:
        uv run fab docs-live
        uv run fab docs-live --port 8080
    """
    docs_dir = os.path.join(os.getcwd(), "docs")
    _clear_docs_build(docs_dir)
    cmd = (
        f"uv run sphinx-autobuild -b dirhtml source build/dirhtml "
        f"--port {port} -v -E -W"
    )
    with c.cd(docs_dir):
        # Unbuffered: the output of the command is piped here, so the messages of
        # sphinx-autobuild (the server URL among them) would stay in its buffer
        c.run(cmd, env={"PYTHONUNBUFFERED": "1"})


# =============================================================================
# Utility Tasks
# =============================================================================


@task
def costs(c: Context, changed: bool = False) -> None:
    """
    Report what the requests recorded for the tests and the evaluations cost.

    The requests in the VCR cassettes are priced from the usage in their
    responses, at the current prices of genai-prices, i.e. what recording them
    again would cost (tests/cassette_costs.py). The costs of the evaluations are
    the ones recorded in their results (dev/evals/costs.py). Sends no request.

    Args:
        changed: Price only the cassettes that differ from the last commit, i.e.
            the ones recorded since, and leave out the evaluations

    Example:
        uv run fab costs
        uv run fab costs --changed
    """
    if changed:
        c.run("uv run python -m tests.cassette_costs --changed")
        return
    c.run("uv run python -m tests.cassette_costs")
    print()
    c.run("uv run python -m dev.evals.costs")


@task(name="install-check")
def install_check(c: Context, version: str = "") -> None:
    """
    Check that ContextGem installs on each platform and supported Python.

    Resolves the dependencies of pyproject.toml, or a release from PyPI, without
    and with their extras, for Linux (glibc and musl), Windows and macOS, on x86_64
    and arm64, from wheels only, as an install without a compiler gets them
    (dev/check_install.py). Takes seconds, and sends no LLM request. Fails when an
    install goes otherwise than the known failures listed there. Run it before
    a release and after a dependency change.

    Args:
        version: A release on PyPI, or "latest" (default: pyproject.toml)

    Example:
        uv run fab install-check
        uv run fab install-check --version latest
    """
    command = "uv run python dev/check_install.py"
    if version:
        command += f" --version {version}"
    result = c.run(command, warn=True)
    if result is None or result.failed:
        raise Exit(code=result.exited if result else 1)


@task(name="docling-fixtures")
def docling_fixtures(
    c: Context,
    write: bool = False,
    outputs: bool = False,
    docling_version: str = "locked",
) -> None:
    """
    Check or write the Docling documents of the test fixtures, with full Docling.

    Converts each file of tests/other_files/docling/ with full Docling, in
    a throwaway environment with the CPU-only PyTorch (dev/docling_fixtures.py),
    and checks that ContextGem makes the same paragraphs from it as from the
    committed Docling document, or writes the Docling documents and the outputs of
    ContextGem from them (<file name>.contextgem.jsonl). Downloads the models of
    Docling on the first run (about 0.5 GB), and sends no LLM request. Run it after
    a Docling upgrade fails the tests, or before a release.

    Args:
        write: Write the Docling documents and the outputs instead of checking them
        outputs: Only write the outputs of ContextGem from the committed Docling
            documents, without Docling: after a change of the converter
        docling_version: "locked" (uv.lock, default), "latest" (the newest from
            before the cooldown of pyproject.toml, `[tool.uv] exclude-newer`) or
            X.Y.Z

    Example:
        uv run fab docling-fixtures
        uv run fab docling-fixtures --write
        uv run fab docling-fixtures --outputs
        uv run fab docling-fixtures --docling-version latest
    """
    command = (
        f"uv run python dev/docling_fixtures.py --docling-version {docling_version}"
    )
    if write:
        command += " --write"
    if outputs:
        command += " --outputs"
    result = c.run(command, warn=True)
    if result is None or result.failed:
        raise Exit(code=result.exited if result else 1)


@task(name="docling-floor")
def docling_floor(c: Context) -> None:
    """
    Run the tests of Document.from_docling() with the oldest docling-core allowed.

    Replaces docling-core with the lower bound of the docling extra, without
    Docling, in a throwaway environment created from uv.lock, and runs the tests of
    Document.from_docling() there, except the one that converts files with Docling
    (dev/docling_floor.py). Takes a minute or two, and sends no LLM request. CI runs
    it on each push.

    Example:
        uv run fab docling-floor
    """
    result = c.run("uv run python dev/docling_floor.py", warn=True)
    if result is None or result.failed:
        raise Exit(code=result.exited if result else 1)


@task
def readme(c: Context) -> None:
    """Regenerate README.md from template."""
    c.run("uv run python dev/populate_project_readme.py")
    print("✓ README.md regenerated from template")


@task(name="readme-preview")
def readme_preview(c: Context, port: int = 9120) -> None:
    """
    Serve README.md as GitHub renders it, to look at before it is pushed.

    GitHub's Markdown API makes the HTML, so what GitHub removes is removed,
    and github-markdown-css gives it GitHub's look (``npm ci`` installs it):
    light on the port, dark on the next one. A reload renders the README again
    when it has changed, one request to GitHub for both pages, two where it has
    alerts: 60 an hour are allowed without a token. The preview shows README.md
    as it is: regenerate it from the template first (``fab readme``).

    Args:
        port: Port of the light page; the dark one is on the next (default: 9120)

    Example:
        uv run fab readme-preview
        uv run fab readme-preview --port 9122
    """
    if not all(
        (_ROOT / _MARKDOWN_CSS.format(scheme=scheme)).is_file()
        for scheme in _PREVIEW_SCHEMES
    ):
        raise Exit(
            "The preview needs github-markdown-css. Install it first: npm ci", code=1
        )
    ports = {scheme: port + offset for offset, scheme in enumerate(_PREVIEW_SCHEMES)}
    if busy := [p for p in ports.values() if _port_in_use(p)]:
        free = busy[-1] + 1
        while any(_port_in_use(free + offset) for offset in range(len(ports))):
            free += 1
        raise Exit(
            f"Port {busy[0]} is in use by another program. Pass a free pair: "
            f"uv run fab readme-preview --port {free}",
            code=1,
        )
    with contextlib.ExitStack() as stack:
        servers = []
        for scheme, scheme_port in ports.items():
            handler = functools.partial(
                _ReadmePreview, scheme=scheme, directory=str(_ROOT)
            )
            # 127.0.0.1, not localhost: on Windows a request to localhost tries
            # IPv6 first and waits two seconds for the refusal.
            address = ("127.0.0.1", scheme_port)
            servers.append(
                stack.enter_context(http.server.ThreadingHTTPServer(address, handler))
            )
            print(
                f"Serving README.md {scheme} at http://127.0.0.1:{scheme_port}/",
                flush=True,
            )
        # The first page is served in this thread, where Ctrl+C reaches it; the
        # others are stopped before their sockets are closed.
        for server in servers[1:]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            stack.callback(server.shutdown)
        print("Ctrl+C stops them", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            servers[0].serve_forever()


@task(name="social-preview")
def social_preview(c: Context, browser: str = "") -> None:
    """
    Render social/github-social-preview.png from the SVG beside it.

    A headless Chrome, Edge or Chromium takes the picture, at 1280 by 640 pixels.
    GitHub reads no file for the preview: upload the picture in the settings of
    the repository.

    Args:
        browser: Path of the browser, where none is found

    Example:
        uv run fab social-preview
        uv run fab social-preview --browser /usr/bin/brave-browser
    """
    browser = browser or _find_browser() or ""
    if not browser:
        raise Exit(
            "No Chrome, Edge or Chromium was found. Name one: "
            "uv run fab social-preview --browser PATH",
            code=1,
        )
    png = _SOCIAL_PREVIEW.with_suffix(".png")
    with tempfile.TemporaryDirectory() as scratch:
        shot = Path(scratch) / png.name
        # The browser takes absolute paths only, and exits with 0 when it
        # wrote nothing. --disable-lcd-text keeps the coloured fringes of
        # sub-pixel text rendering out of the picture.
        result = c.run(
            f'"{browser}" --headless --disable-lcd-text --hide-scrollbars'
            " --force-device-scale-factor=1 --window-size=1280,640"
            f' "--screenshot={shot}" "{_SOCIAL_PREVIEW.as_uri()}"',
            hide=True,
            warn=True,
        )
        if not shot.is_file():
            stderr = result.stderr if result else ""
            raise Exit(f"{browser} wrote no image.\n{stderr}", code=1)
        shutil.copyfile(shot, png)
    print(f"✓ Wrote {png.relative_to(_ROOT).as_posix()}")
