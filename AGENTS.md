# AGENTS.md - AI Coding Assistant Guidelines

Guidelines for AI coding assistants working with the ContextGem codebase.

For detailed contribution procedures, see `CONTRIBUTING.md`, which covers:

- Development environment setup
- Project structure overview
- VCR cassette recording scenarios
- Documentation building

## Project Overview

**ContextGem** is a Python LLM framework for extracting structured data from documents using long context windows (not RAG-based).

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
| Required import | `from __future__ import annotations` (except `__init__.py` and the files under `dev/`, e.g. the usage examples) |
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
2. **Update RST files** in `docs/source/` if the feature has dedicated documentation
3. **Update `dev/readme.template.md`** if README content is affected (not `README.md` directly), and `PYPI.md` if the summary, the install command or the feature list changes
4. **Add usage examples** to `dev/usage_examples/` if demonstrating new features, and list each new example in `tests/test_usage_examples.py`: in `_EXAMPLES_WITHOUT_REQUESTS` if it sends no requests, or in `_EXAMPLES_WITH_REQUESTS` if it does, in which case its test is a VCR test whose cassette the user has to record (**do NOT run it**). A change to the requests of an existing example invalidates the cassette of that example only
5. **Verify docs compile** by running from the `docs/` directory:

   ```bash
   uv run sphinx-build -b dirhtml source build/dirhtml -v -E -W
   ```

### VCR Cassette Rules

- **Never run tests in "live" mode** without existing cassettes
- Tests replay recorded LLM API responses from `tests/cassettes/`
- Cassette files are named `<TestClass>.<test name>.yaml`: when moving a VCR test to another test class, rename its cassettes with `git mv` in the same change (a test whose cassette is missing records live)
- If tests fail due to cassette mismatches, **inform the user** that cassettes need re-recording
- The user will handle cassette re-recording themselves (requires API keys and may incur costs)
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
uv run pre-commit run --all-files       # Run all linters/formatters
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
6. **Never edit version fields by hand** - `cz bump` owns `pyproject.toml:version`, `contextgem/__init__.py:__version__`, `CITATION.cff:version`, and `[tool.commitizen].version`. `CHANGELOG.md` is hand-maintained, so writing the next version's section there is fine: each version has a compare link at the end of the file, newest first, which `tests/test_changelog.py` checks.
7. **A release is dated by hand** - `cz bump` does not date it: write the release date into the changelog heading of the new version and into `CITATION.cff:date-released` before the bump. The last of the commitizen `pre_bump_hooks` runs `tests/test_changelog.py`, which stops a bump whose two dates differ. Give `cz bump` the version for a major release (`uv run cz bump 2.0.0`): from the commit types alone it proposes a minor one.
8. **GitHub Actions SHA pinning: use the commit SHA, not the tag-object SHA** - some action repos (e.g. `github/codeql-action`) use annotated tags. `git/refs/tags/<tag>` returns the tag-object SHA for those, which is *not* what `uses:` should pin to. Resolve via `repos/{owner}/{repo}/commits/<tag>` (auto-dereferences) or `git/tags/<sha>` to get the underlying commit SHA.
9. **`dev/evals/` sends live requests** - the evaluation harness measures extraction quality and cost on evaluation cases against a recorded baseline. Run its live scripts only when asked, and never edit its recorded results. See `dev/evals/AGENTS.md`.
10. **`README.md` is for GitHub, `PYPI.md` is for PyPI** - `PYPI.md` is the package description (`project.readme` in `pyproject.toml`): short, hand-maintained, without images, with absolute HTTPS links only. In `dev/readme.template.md`, take images by relative paths from `docs/source/_static/` (shared with the documentation site) or `assets/` (the README's own), so that they render in a local preview and on any branch, and link to the files of the repository by relative paths (`CONTRIBUTING.md`). Never delete an image of `docs/source/_static/` that a released README loads from `https://contextgem.dev/_static/`. `social/` holds the repository's social preview: the SVG it is drawn in and the PNG rendered from it (`uv run fab social-preview`), which is uploaded by hand in the settings of the repository; it carries nothing that a release changes.
11. **The checks read their versions from `uv.lock`** - the pre-commit hooks run the tools through `uv run`, CI runs the same hooks (`uv run pre-commit run --all-files`) after `uv sync --locked`, and the development Python is the one in `.python-version`. To change the version of a tool, change `uv.lock`, not the hook configuration. The hooks for trailing whitespace and the ends of files skip `contextgem/`, `tests/`, `dev/usage_examples/` and `dev/evals/`, whose bytes go into the requests that the VCR cassettes match.
