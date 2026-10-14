"""
Fabric automation tasks for ContextGem development.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fabric import task
from invoke.context import Context
from invoke.exceptions import Exit


_ROOT = Path(__file__).parent

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


# =============================================================================
# Development Setup Tasks
# =============================================================================


@task
def sync(c: Context) -> None:
    """Upgrade dependencies within pyproject.toml constraints (rewrites uv.lock)."""
    c.run("uv sync --all-groups --upgrade")


@task
def setup(c: Context) -> None:
    """Set up development environment (install locked deps + hooks)."""
    print("Installing dependencies...")
    # The versions locked in uv.lock, i.e. the ones that CI uses: no upgrades
    c.run("uv sync --all-groups")
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


@task
def docs(c: Context) -> None:
    """Build documentation (static build with full rebuild)."""
    docs_dir = os.path.join(os.getcwd(), "docs")
    cmd = "uv run sphinx-build -b dirhtml source build/dirhtml -v -E -W"
    with c.cd(docs_dir):
        c.run(cmd)
    print("\n✓ Documentation built successfully!")
    print("  To view, serve the folder and open http://localhost:9000:")
    print("  uv run python -m http.server 9000 --directory docs/build/dirhtml")


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
def readme(c: Context) -> None:
    """Regenerate README.md from template."""
    c.run("uv run python dev/populate_project_readme.py")
    print("✓ README.md regenerated from template")


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
