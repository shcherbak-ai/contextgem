# AGENTS.md - AI Coding Assistant Guidelines

Guidelines for AI coding assistants working with the ContextGem codebase.

For detailed contribution procedures, see `CONTRIBUTING.md`, which covers:

- Development environment setup
- Project structure overview
- VCR cassette recording scenarios
- Documentation building

## Project Overview

**ContextGem** is a Python framework for extracting structured data and insights from documents using LLMs and decision models, with long context windows (not RAG-based).

- **Python**: 3.11-3.14
- **License**: Apache 2.0
- **Package Manager**: `uv`

## Architecture: Internal/Public Split

The codebase uses a two-layer architecture. **Always implement in internal first, then expose via public.**

```text
contextgem/
├── internal/          # Core implementation (_underscore-prefixed classes)
│   ├── base/          # Business logic (concepts, aspects, documents, llms)
│   ├── prompts/       # Jinja2 prompt templates
│   ├── typings/       # Type system & validators
│   └── registry.py    # Internal-to-public type mapping
└── public/            # Thin facades (inherit from internal, registered via decorator)
```

### Pattern Example

```python
# 1. Internal implementation (contextgem/internal/base/concepts.py)
class _StringConcept(BaseModel):
    name: str
    # ... business logic ...

# 2. Public facade (contextgem/public/concepts.py)
@_expose_in_registry(additional_key=_StringConcept)
class StringConcept(_StringConcept):
    """Public API documentation."""
    pass

# 3. Export in contextgem/__init__.py
```

## Coding Conventions

| Convention | Rule |
|------------|------|
| Internal classes | `_Aspect`, `_Document` (underscore prefix) |
| Public classes | `Aspect`, `Document` (no prefix) |
| Constants | `_MAX_NESTING_LEVEL` (ALL_CAPS) |
| Internal imports | From the module that defines the name: `from contextgem.internal.base.concepts import _StringConcept`. The `__init__.py` files under `contextgem/internal/` stay empty; only `contextgem/__init__.py` and `contextgem/public/__init__.py` import names to give them a shorter path |
| Required import | `from __future__ import annotations` (except `__init__.py` and the files under `dev/`, e.g. the usage examples) |
| File header | `# SPDX-FileCopyrightText: Shcherbak AI AS` and `# SPDX-License-Identifier: Apache-2.0` as the first two lines of every Python file, except empty `__init__.py` files and the usage examples (`dev/usage_examples/`), which the documentation shows; documentation pages have no header. The `add-file-headers` pre-commit hook adds the lines where they are missing |
| Formatter | Ruff (line length: 88) |
| Type checker | ty |
| Docstrings | reStructuredText format for Sphinx (`:param:`, `:type:`, `:ivar:`, `:vartype:`, `:returns:`, `:rtype:`) |
| Data models | Pydantic v2 (`BaseModel`, `field_validator`, `model_validator`) |

## Auto-Generated Files - Do NOT Edit

| File | Source | Regeneration |
|------|--------|--------------|
| `README.md` | `dev/readme.template.md` | Pre-commit hook runs `dev/populate_project_readme.py` |
| `uv.lock` | `pyproject.toml` | `uv sync` |

**To update README**: Edit `dev/readme.template.md`, then run pre-commit or the generation script.

## After Code Changes

**Always run these steps after making changes:**

```bash
# 1. Run pre-commit hooks (formatting, linting, type checking)
uv run pre-commit run --all-files

# 2. Run targeted tests for the code you modified
uv run pytest tests/test_extraction.py::TestExtraction::test_relevant_method

# 3. Run full test suite to check for regressions. If you added tests that call
# LLM APIs, deselect them: a test whose cassette is missing records live
uv run pytest
# uv run pytest --deselect "tests/test_<area>.py::Test<Area>::test_llm_feature"
```

### Writing Tests for New Implementations

When adding new functionality:

1. **Add test methods** to the test class of the relevant feature area (`tests/test_<area>.py::Test<Area>`, e.g. `tests/test_extraction.py::TestExtraction`) following existing patterns. The test classes inherit the shared test objects (documents, LLMs, images) from `tests/base.py::BaseTest`. A test must leave them unchanged: a test that changes a document works on its copy (`document = self.document.clone()`)
2. **For tests that do NOT call LLM APIs**: Run them to verify they pass
3. **For tests that call LLM APIs**: Add `@pytest.mark.vcr` decorator, **do NOT run them**, and inform the user that cassettes need recording before these tests can be executed

```python
# Example: Adding a test in tests/test_extraction.py
class TestExtraction(BaseTest):
    def test_non_llm_feature(self):
        # Safe to run - no LLM calls
        pass

    @pytest.mark.vcr  # Requires cassette - DO NOT RUN, inform user
    def test_llm_feature(self):
        # Calls LLM API - user must record cassette first
        pass
```

### Updating Documentation

When adding or modifying functionality:

1. **Update docstrings** in the affected classes/methods (reStructuredText format)
2. **Update RST files** in `docs/source/` if the feature has dedicated documentation. A new page starts with a `.. meta::` block whose `:description:` says in one sentence what the page covers: it is the page's line in `llms.txt` and its meta description (the `docs-page-description` hook checks that it is there)
3. **Update `dev/readme.template.md`** if README content is affected (not `README.md` directly), and `PYPI.md` if the summary, the install command or the feature list changes. Then look at the README in a browser (below)
4. **Add usage examples** to `dev/usage_examples/` if demonstrating new features, and list each new example in `tests/test_usage_examples.py`: in `_EXAMPLES_WITHOUT_REQUESTS` if it sends no requests, or in `_EXAMPLES_WITH_REQUESTS` if it does, in which case its test is a VCR test whose cassette the user has to record (**do NOT run it**). A change to the requests of an existing example invalidates the cassette of that example only
5. **Verify docs compile** by running from the `docs/` directory:

   ```bash
   uv run sphinx-build -b dirhtml source build/dirhtml -v -E -W
   ```

### Checking the README in a browser

- **A change to the README (through `dev/readme.template.md`), or to an image it shows, isn't done until the README was looked at as GitHub renders it**: at 1280 pixels wide (`browser_resize` to 1280x900) and at 390 (390x844), in light and in dark (the preview's two pages), with the figure played to its end. Say what you looked at, and what you couldn't check.
- **`uv run fab readme-preview` serves `README.md` in light at `http://127.0.0.1:9120/` and in dark at `http://127.0.0.1:9121/`**, each page in its theme whatever the browser's colour scheme. It shows the file as it is, so regenerate it from the template first (`uv run fab readme`). Ask the first address and use both if its title is `README.md · contextgem · light`, else start the task in the background (`--port N`, which serves them at N and N+1, where another program holds a port: licenseal's preview uses 9100, tethered's 9110 and 9111). What the page shows, and how often it asks GitHub, is in [CONTRIBUTING.md](CONTRIBUTING.md#-updating-readmemd).
- **Use the `playwright` browser** ([Playwright MCP](https://github.com/microsoft/playwright-mcp)). `.mcp.json` starts it, in the version that `package-lock.json` holds (`npm ci` installs it), once the developer has approved it: the installed Chrome, without a window, with a fresh profile for each session. Drive it a step at a time with its tools (`browser_navigate`, `browser_resize`, `browser_emulate_media`, `browser_take_screenshot`, `browser_evaluate`), and don't write Playwright scripts of your own.
  - It refuses addresses other than `localhost`, `127.0.0.1`, and `camo.githubusercontent.com`, GitHub's image proxy, which serves the badges (`--allowed-origins`). That catches a mistake and isn't a security boundary.
- **The figure plays for a few seconds after the page loads**, and the gem of the wordmark shines once, so a screenshot taken at once catches them part-way: wait, then take it. Emulated reduced motion doesn't reach an image inside `<img>`. To check that the figure stops, open the SVG itself (`http://127.0.0.1:9120/docs/source/_static/contextgem_readme_extraction.svg`) with `reducedMotion: "reduce"`: it shows the whole figure at once.
- **Everything the browser saves goes in `.playwright-mcp/`**, which git ignores, and nothing in the project's root. It puts the files it names itself there; a file that you name is taken from the root, so name it with the folder: `.playwright-mcp/readme-390.png`. In Claude Code a hook refuses any other name (`.claude/hooks/playwright-files.mjs`, which `.claude/settings.json` starts before each of the browser's tools; pre-commit runs its tests).
- **Clear out your own files when the work is done.** Delete what your session put in `.playwright-mcp/`, never the folder as a whole: another session may be using it. A screenshot that your last message asks the developer to look at stays, named in that message.

### VCR Cassette Rules

- **Never run tests in "live" mode** without existing cassettes
- Tests replay recorded LLM API responses from `tests/cassettes/`
- Cassette files are named `<TestClass>.<test name>.yaml`: when moving a VCR test to another test class, rename its cassettes with `git mv` in the same change (a test whose cassette is missing records live)
- If tests fail due to cassette mismatches, **inform the user** that cassettes need re-recording
- The user will handle cassette re-recording themselves (requires API keys and may incur costs)
- **What recording costs is computed offline, never by recording**: `uv run fab costs` prices the usage in the recorded responses at the current prices (`--changed` for the cassettes that differ from the last commit), and adds up the costs recorded in the results of `dev/evals/`. A test run reports at its end what the cassettes it recorded cost, and what those it replayed would cost if sent live
- See [CONTRIBUTING.md - VCR Cassette Management](CONTRIBUTING.md#-vcr-cassette-management) for scenarios

## Git Policy

**Never stage or commit changes** - this is the developer's responsibility.

The developer will review changes and handle git operations themselves.

### File Operations During Refactoring

When moving or renaming files, **always use `git mv`** to preserve git history:

```bash
# Moving a file
git mv old/path/file.py new/path/file.py

# Renaming a file
git mv old_name.py new_name.py
```

**Never** use regular file system operations (copy + delete, or IDE rename) for moves/renames - this breaks git history tracking.

## Quick Commands

```bash
uv sync --all-groups                    # Install dependencies
npm ci                                  # Install the development tools of package.json
uv run pre-commit run --all-files       # Run all linters/formatters
uv run fab readme-preview               # Serve README.md as GitHub renders it
uv run fab costs                        # What the recorded requests of the tests and evals cost
uv run fab install-check                # Whether the dependencies install on each platform and Python
uv run fab docling-fixtures              # Check the Docling documents of the test fixtures with full Docling
uv run pytest                           # Run tests (uses recorded cassettes)
uv run pytest --cov=contextgem          # Run tests with coverage
uv run pytest tests/test_extraction.py::TestExtraction::test_specific  # Run specific test
```

## Key Gotchas

1. **Never import public classes in internal modules** - use registry for type resolution
2. **Prompt changes break VCR cassettes** - inform user if tests fail after prompt modifications
3. **README.md is auto-generated** - edit `dev/readme.template.md` instead
4. **Never stage or commit** - let the developer handle all git operations
5. **Always run pre-commit** after code changes before considering work complete
6. **Never edit version fields by hand** - `cz bump` owns `pyproject.toml:version`, `contextgem/__init__.py:__version__`, `CITATION.cff:version`, and `[tool.commitizen].version`. `CHANGELOG.md` is hand-maintained: the changes of the next release go under `## [Unreleased]`, whose compare link ends at `HEAD`. Each version has a compare link at the end of the file, newest first, which `tests/test_changelog.py` checks.
7. **A release is dated by a hook of `cz bump`** - `dev/date_release.py`, one of the commitizen `pre_bump_hooks`, renames `## [Unreleased]` to the new version with the day of the bump (the local date), points its compare link at the new tag and writes the day into `CITATION.cff:date-released`; the bump commits both files. When the newest section is the released version's, it stops the bump. The last hook runs `tests/test_changelog.py`, which stops a bump whose two dates differ. A stopped bump has made no commit or tag, but leaves the version fields rewritten and `uv.lock` updated and staged, and the release dated if the dating hook ran: set the version fields back and restore `uv.lock` before running it again, or commitizen finds nothing to bump. On the next try the dating hook names and dates its section again, for whatever version that bump makes. Give `cz bump` the version for a major release (`uv run cz bump 2.0.0`): from the commit types alone it proposes a minor one. Before the bump, run `uv run fab install-check` (see 13).
8. **GitHub Actions SHA pinning: use the commit SHA, not the tag-object SHA** - some action repos (e.g. `github/codeql-action`) use annotated tags. `git/refs/tags/<tag>` returns the tag-object SHA for those, which is *not* what `uses:` should pin to. Resolve via `repos/{owner}/{repo}/commits/<tag>` (auto-dereferences) or `git/tags/<sha>` to get the underlying commit SHA.
9. **`dev/evals/` sends live requests** - the evaluation harness measures extraction quality and cost on evaluation cases against a recorded baseline. Run its live scripts only when asked, and never edit its recorded results. See `dev/evals/AGENTS.md`.
10. **`README.md` is for GitHub, `PYPI.md` is for PyPI** - `PYPI.md` is the package description (`project.readme` in `pyproject.toml`): short, hand-maintained, without images, with absolute HTTPS links only. In `dev/readme.template.md`, take images by relative paths from `docs/source/_static/` (shared with the documentation site) or `assets/` (the README's own), so that they render in a local preview and on any branch, and link to the files of the repository by relative paths (`CONTRIBUTING.md`). Never delete an image of `docs/source/_static/` that a released README loads from `https://contextgem.dev/_static/`. `social/` holds the repository's social preview: the SVG it is drawn in and the PNG rendered from it (`uv run fab social-preview`), which is uploaded by hand in the settings of the repository, and which the documentation site shows as the picture of its links (`html_extra_path` and `ogp_image` in `docs/source/conf.py`). It carries nothing that a release changes; when the wordmark, the README's heading or the tagline changes, change the SVG with it, render the PNG again and upload it.
11. **The checks read their versions from `uv.lock`** - the pre-commit hooks run the tools through `uv run`, CI runs the same hooks (`uv run pre-commit run --all-files`) after `uv sync --locked`, and the development Python is the one in `.python-version`. To change the version of a tool, change `uv.lock`, not the hook configuration. The hooks for trailing whitespace and the ends of files skip `contextgem/`, `tests/`, `dev/usage_examples/` and `dev/evals/`, whose bytes go into the requests that the VCR cassettes match.
12. **Only `litellm`, `genai-prices` and `openai` are pinned** - every other dependency, runtime or dev, has a range that admits its latest release, bounded at the next major where it has an upper bound (`<1.0.0` for a 0.x tool), so that `uv lock --upgrade` finds the latest release. Never cap or pin a tool to avoid its findings: when ruff or ty is upgraded, make the code pass their checks, with a `noqa` or `ty: ignore` and a reason only where a pattern is deliberate. `openai` goes only as far as the pinned `litellm` allows.
13. **Check the installs before a release and after a dependency change** - the tests run on the locked versions, on the platform they run on. `uv run fab install-check` resolves the dependencies of `pyproject.toml`, with its extras, for Linux (glibc and musl), Windows and macOS, on x86_64 and arm64, and each supported Python, from wheels only, as an install without a compiler gets them: in seconds, without a request to a model. `--version latest` checks the release on PyPI. It fails on any difference from the installs known to fail (`EXPECTED_FAILURES` in `dev/check_install.py`): a new failure, e.g. a new pin of `litellm` without the wheels of a platform, or a gap that closed. Update the list, and the documentation that names the gap, e.g. the note on Alpine in `docs/source/installation.rst`.
14. **The `docling` extra installs Docling for `Document.from_docling()`** - Docling and its document model, `docling-core`, whose floor is the API that the converter uses. Only `contextgem/internal/converters/docling.py` imports `docling_core`, and only `from_docling()` imports that module, so that ContextGem imports without the extra. The module checks the installed version of docling-core first, against `_MIN_DOCLING_CORE_VERSION`, as it can be installed without the extra, e.g. with an older Docling: raise the floor there too (`test_from_docling_without_docling_core` checks both, and that the floor matches `pyproject.toml`). Full Docling runs on PyTorch, which is not in the dev group: the dev group has `docling-core` (same range as the extra) and the torch-free `docling-slim`, with which the tests convert the DOCX, XLSX, PPTX, HTML and Markdown fixtures of `tests/other_files/docling/` (files written for ContextGem, and test documents of Docling in `docling_tests/`, under its MIT License: add only files that may be redistributed). A PDF fixture is read from the Docling document committed next to it; the usage examples that convert a PDF run with a stand-in for its converter (`tests/test_usage_examples.py`). When an upgrade of Docling changes the output, the tests of the fixtures fail: check the difference with `uv run fab docling-fixtures`, which uses full Docling in a throwaway environment, and write the Docling documents with `--write` (the workflow `docling-fixtures.yml` runs the check in CI). Full Docling brings NVIDIA's CUDA libraries on Linux, which `licenseal.review.toml` lists as reviewed: their versions follow torch, so a torch upgrade needs new entries (`uv run licenseal init-review-file --merge`, then fill in the license and the note).
