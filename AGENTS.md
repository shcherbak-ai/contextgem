# AGENTS.md - AI Coding Assistant Guidelines

Guidelines for AI coding assistants working with the ContextGem codebase.

For detailed contribution procedures, see `CONTRIBUTING.md`, which covers:
- Development environment setup
- Project structure overview
- VCR cassette recording scenarios
- Documentation building

## Project Overview

**ContextGem** is a Python LLM framework for extracting structured data from documents using long context windows (not RAG-based).

- **Python**: 3.10-3.13
- **License**: Apache 2.0
- **Package Manager**: `uv`

## Architecture: Internal/Public Split

The codebase uses a two-layer architecture. **Always implement in internal first, then expose via public.**

```
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
| Required import | `from __future__ import annotations` (except `__init__.py`) |
| Formatter | Ruff (line length: 88) |
| Type checker | Pyright (basic mode) |
| Docstrings | reStructuredText format for Sphinx (`:param:`, `:type:`, `:ivar:`, `:vartype:`, `:returns:`, `:rtype:`) |
| Data models | Pydantic v2 (`BaseModel`, `field_validator`, `model_validator`) |

## Auto-Generated Files - Do NOT Edit

| File | Source | Regeneration |
|------|--------|--------------|
| `README.md` | `dev/readme.template.md` | Pre-commit hook runs `dev/populate_project_readme.py` |
| `docs/source/llms.txt` | Documentation sources | Pre-commit hook |
| `dev/notebooks/` | `dev/usage_examples/` | Pre-commit hook |
| `uv.lock` | `pyproject.toml` | `uv sync` |

**To update README**: Edit `dev/readme.template.md`, then run pre-commit or the generation script.

## After Code Changes

**Always run these steps after making changes:**

```bash
# 1. Run pre-commit hooks (formatting, linting, type checking)
uv run pre-commit run --all-files

# 2. Run targeted tests for the code you modified
uv run pytest tests/test_all.py::TestAll::test_relevant_method

# 3. Run full test suite to check for regressions
uv run pytest
```

### Writing Tests for New Implementations

When adding new functionality:

1. **Add test methods** to `tests/test_all.py::TestAll` following existing patterns
2. **For tests that do NOT call LLM APIs**: Run them to verify they pass
3. **For tests that call LLM APIs**: Add `@pytest.mark.vcr` decorator, **do NOT run them**, and inform the user that cassettes need recording before these tests can be executed

```python
# Example: Adding a test in tests/test_all.py
class TestAll:
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
3. **Update `dev/readme.template.md`** if README content is affected (not `README.md` directly)
4. **Add usage examples** to `dev/usage_examples/` if demonstrating new features
5. **Verify docs compile** by running from the `docs/` directory:

   ```bash
   uv run sphinx-build -b dirhtml source build/dirhtml -v -E -W
   ```

### VCR Cassette Rules

- **Never run tests in "live" mode** without existing cassettes
- Tests replay recorded LLM API responses from `tests/cassettes/`
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
uv run pytest tests/test_all.py::TestAll::test_specific  # Run specific test
```

## Key Gotchas

1. **Never import public classes in internal modules** - use registry for type resolution
2. **Prompt changes break VCR cassettes** - inform user if tests fail after prompt modifications
3. **README.md is auto-generated** - edit `dev/readme.template.md` instead
4. **Never stage or commit** - let the developer handle all git operations
5. **Always run pre-commit** after code changes before considering work complete
