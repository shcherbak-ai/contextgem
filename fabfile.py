"""
Fabric automation tasks for ContextGem development.
"""

from __future__ import annotations

import os

from fabric import task
from invoke.context import Context


# =============================================================================
# Development Setup Tasks
# =============================================================================


@task
def sync(c: Context) -> None:
    """Sync dependencies from pyproject.toml with upgrades."""
    c.run("uv sync --all-groups --upgrade")


@task
def setup(c: Context) -> None:
    """Set up development environment (install deps + hooks)."""
    print("Installing dependencies...")
    sync(c)
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
    print("  Open docs/build/dirhtml/index.html to view")


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
        c.run(cmd)


# =============================================================================
# Utility Tasks
# =============================================================================


@task
def readme(c: Context) -> None:
    """Regenerate README.md from template."""
    c.run("uv run python dev/populate_project_readme.py")
    print("✓ README.md regenerated from template")
