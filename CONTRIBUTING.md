# 🤝 Contributing to ContextGem

Thank you for your interest in contributing to ContextGem! This document provides guidelines and instructions for contributing to the project.

## 📋 Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the expectations for all interactions within the project.

## ✍️ Contributor Agreement

Before your contribution can be accepted, you must sign our [Contributor Agreement](/.github/CONTRIBUTOR_AGREEMENT.md). This is a legal document that grants us the necessary rights to use your contribution. The agreement is based on the [Oracle Contributor Agreement](https://www.oracle.com/technetwork/oca-405177.pdf) and this requirement follows [OpenSSF Best Practices](https://www.bestpractices.dev/en) for silver-level criteria (section "Project oversight").

To sign the agreement:

1. Read the [Contributor Agreement](/.github/CONTRIBUTOR_AGREEMENT.md) carefully
2. Create a copy of the agreement under `.github/contributors/[your-github-username].md`
3. Fill in all the requested information and include it in your first pull request

## 🤖 Using AI Coding Assistants

This repository is **AI agent-friendly**: **[AGENTS.md](AGENTS.md)** gives AI coding assistants the project overview, architecture patterns, coding conventions, and workflow guidelines ([agents.md standard](https://agents.md)), and a directory with rules of its own has an `AGENTS.md` too (e.g. `dev/evals/`). Claude Code, Codex, Cursor and other assistants read these files.

When using AI assistants (Claude Code, Cursor, etc.) to contribute:

1. **Review AI-generated code** - Always verify changes follow project patterns and pass tests
2. **Handle VCR cassettes yourself** - AI assistants should not run tests that call LLM APIs without existing cassettes
3. **Manage git operations yourself** - Review and commit changes manually rather than letting AI handle git

> **💡 Tip:** AI assistants work best when given specific, focused tasks. Break large contributions into smaller pieces for better results.

## 🚀 Getting Started

### 🛠️ Development Environment

1. **🍴 Fork and clone the repository**:

    - First, fork the repository by clicking the "Fork" button on the GitHub project page

    - Then clone your fork to your local machine:

    ```bash
    git clone https://github.com/YOUR-GITHUB-USERNAME/contextgem.git
    cd contextgem
    ```

    - Add the original repository as an upstream remote:

    ```bash
    git remote add upstream https://github.com/shcherbak-ai/contextgem.git
    ```

2. **⚙️ Set up the development environment**:

    ```bash
    # Install uv if you don't have it:
    # https://docs.astral.sh/uv/getting-started/installation/

    # Install dependencies, as locked in uv.lock and package-lock.json, and
    # pre-commit hooks. The Python version of the environment is the one in
    # .python-version
    uv run fab setup

    # Or manually:
    # uv sync --all-groups
    # npm ci
    # uv run pre-commit install
    # uv run pre-commit install --hook-type commit-msg
    ```

    The development tools that `package.json` lists need [Node.js](https://nodejs.org/): the stylesheet of the README preview and the browser of AI coding assistants. None of them is part of the package.

### 🛠️ Available Fabric Commands

The project includes a `fabfile.py` with common development tasks:

```bash
uv run fab --list          # List all available commands
uv run fab setup           # Set up dev environment (locked deps, npm tools, hooks)
uv run fab sync            # Upgrade dependencies (rewrites uv.lock)
uv run fab lint            # Run pre-commit checks on all files
uv run fab docs            # Build documentation
uv run fab docs-live       # Start live documentation server
uv run fab readme          # Regenerate README.md from template
uv run fab readme-preview  # Serve README.md as GitHub renders it
uv run fab social-preview  # Render social/github-social-preview.png from its SVG
uv run fab costs           # Report what the recorded requests of the tests and evals cost
uv run fab install-check   # Check that the dependencies install on each platform and Python
uv run fab docling-fixtures # Check the Docling documents of the test fixtures with full Docling
uv run fab docling-floor   # Run the tests of Document.from_docling() with the oldest docling-core allowed
uv run fab install-hooks   # Install pre-commit hooks
```

### 📁 Project Structure

Below is a high-level overview of the codebase layout and where to make different types of contributions:

```text
contextgem/
│
├── contextgem/
│   │
│   ├── internal/                 # 🔧 Core implementation (start here for new features)
│   │   ├── base/                 #   - Core abstractions & business logic
│   │   │   ├── concepts.py       #     - Internal concept implementations
│   │   │   ├── aspects.py        #     - Internal aspect implementations
│   │   │   ├── documents.py      #     - Internal document processing
│   │   │   ├── llms.py           #     - Internal LLM functionality
│   │   │   ├── models.py         #     - Internal model processing logic & model groups
│   │   │   └── ...               #     - More internal implementations
│   │   ├── prompts/              #   - LLM prompt templates
│   │   ├── typings/              #   - Type definitions
│   │   └── ...                   #   - More internal modules
│   │
│   └── public/                   # 🎯 User-facing API (thin facades exposing internals)
│       ├── concepts.py           #   - Public concept facades
│       ├── aspects.py            #   - Public aspect facades
│       ├── documents.py          #   - Public document facades
│       ├── pipelines.py          #   - Public pipeline facades
│       ├── llms.py               #   - Public LLM facades
│       ├── models.py             #   - Public model group facade
│       └── ...                   #   - More public modules
│
├── tests/
│   ├── cassettes/                # 📼 VCR recordings (auto-generated)
│   ├── base.py                   # 🧱 Base class of feature/integration tests (shared test objects)
│   ├── cassette_costs.py         # 💰 Prices the requests recorded in the cassettes
│   ├── test_<area>.py            # ✅ Feature/integration tests, one module per feature area
│   ├── test_properties.py        # 🎲 Hypothesis property tests for pure helpers
│   ├── test_units.py             # 🔬 Example-based unit tests for pure helpers
│   ├── utils.py                  # 🛠️ Test utilities & dummy env vars
│   └── ...                       # 📁 Test data files
│
├── docs/
│   ├── source/                   # 📚 Documentation source files
│   └── ...                       # 📋 Build configs & outputs
│
├── dev/
│   ├── usage_examples/           # 📝 Code examples for docs
│   ├── evals/                    # 📊 Extraction evaluation harness, cases & baseline results
│   ├── readme.template.md        # ✏️ Edit this, not README.md
│   └── ...                       # 🛠️ Development scripts
│
├── assets/                       # 🖼️ Images of the README only
├── pyproject.toml                # ⚙️ Dependencies & project config
├── package.json                  # 🧰 Development tools from npm (README preview, agents' browser)
├── PYPI.md                       # 📦 Package description shown on PyPI
└── README.md                     # 🤖 Auto-generated (don't edit)
```

**🎯 Quick Start for Your Contribution:**

- **Adding new functionality?** → Implement it in `contextgem/internal/` (base classes, validation, serialization, typing). Then expose it through a thin, documented facade in `contextgem/public/` that inherits from the internal class and is registered with the `@_expose_in_registry` decorator, so that deserialization and instance creation return public types. Do not import public classes in internal modules: use the registry to resolve types. Import an internal name from the module that defines it: the `__init__.py` files under `contextgem/internal/` stay empty.
- **Writing tests?** → Most tests go to the module of the relevant feature area (`tests/test_<area>.py`, e.g. `tests/test_extraction.py::TestExtraction`). See [Testing](#-testing) for the others.
- **Updating docs?** → Edit the pages in `docs/source/`, and the code they show in `dev/usage_examples/`
- **Fixing README?** → Edit `dev/readme.template.md`

---

### ✏️ Making Changes

1. **🌿 Create a new branch** from the latest `dev` branch of the original repository. Pull requests are merged into `dev`, while `main`, the branch of a fresh clone, is behind it:

    ```bash
    git fetch upstream
    git checkout -b feature/your-feature-name upstream/dev
    ```

    When creating a branch, use one of the following prefixes that matches your change type:

    - `bugfix/` - For bug fixes (e.g., `bugfix/fix-llm-timeout`)
    - `feature/` - For new features (e.g., `feature/add-new-concept-type`)
    - `breaking/` - For breaking changes (e.g., `breaking/concepts-api-v2`)
    - `docs/` - For documentation updates (e.g., `docs/update-aspects-guide`)
    - `perf/` - For performance improvements (e.g., `perf/optimize-prompts`)
    - `refactor/` - For code cleanup or refactoring (e.g., `refactor/simplify-error-handling`)

    General guidelines:
    - Use hyphens (-) between words, not underscores or spaces
    - Be specific but concise about what the branch does
    - Include issue numbers when applicable (e.g., `bugfix/issue-42`)
    - Keep branch names lowercase

2. **📝 Make your changes** following the conventions in [AGENTS.md](AGENTS.md#coding-conventions). Write docstrings in reStructuredText (`:param:`, `:type:`, `:returns:`, `:rtype:`, ...): the API reference of the documentation is built from them.

    The pre-commit hooks check and format your code when you commit:

    - **Ruff**: formatting and linting
    - **ty**: static type checking
    - **Bandit**: security issues in the Python code
    - **Interrogate**: docstring coverage, which must stay at 100%: every module, class and function has a docstring
    - **Deptry**: the declared dependencies against the imported ones (unused, missing, transitive)
    - **markdownlint-cli2**: Markdown style
    - **add-file-headers**: adds the license header (two SPDX lines) to a Python file that has none, except a usage example
    - **docs-page-description**: each page of the documentation starts with a meta description (see [Documentation Structure](#-documentation-structure))
    - **update-readme**: regenerates `README.md` from its template (see [Updating README.md](#-updating-readmemd))
    - **package-check**: builds the package and validates its metadata (`uv build` + `twine check`)
    - **uv lock check**: `uv.lock` is in sync with `pyproject.toml`
    - **latest-versions**: the ranges of `pyproject.toml` and `package.json` admit the latest release of each dependency, and `uv.lock` and `package-lock.json` hold the newest versions that the ranges admit, but for the packages listed with their reasons in `EXCEPTIONS` in [dev/check_latest_versions.py](dev/check_latest_versions.py), e.g. the pinned `litellm`, of whose newer releases it warns without failing. It asks PyPI and npm at each commit, as a release can come out at any time, so it can fail on a change that touches no dependency: commit with `SKIP=latest-versions` in front of the command, and the maintainer upgrades the dependencies
    - **Commitizen**: the format of the commit message (see below)
    - **claude hooks tests**: the tests of the Claude Code hook in `.claude/hooks/` and of the settings that start it, with `node --test`
    - the checks of YAML and TOML files, trailing whitespace and the ends of files. The last two skip the files whose bytes go into the requests recorded in the VCR cassettes (the package, the tests, the usage examples and the evaluation cases) and the signed contributor agreements.

    The hooks run the tools through `uv run`, at the versions locked in `uv.lock`, and CI runs the same hooks (`uv run pre-commit run --all-files`), so a check that passes locally passes in CI. CI skips latest-versions, whose result depends on the day rather than on the commit. CI also checks the license compatibility of the dependencies, with licenseal.

    - **If a hook modifies files** (e.g. Ruff formatting, the README): review the changes, add the modified files to the staging area, and commit again.
    - **If a hook reports a problem** (e.g. a Bandit finding or a type error): fix it, add the fixed files to the staging area, and commit again.

3. **🧪 Run tests** to ensure your changes do not break existing functionality:

   ```bash
   uv run pytest
   ```

   > **Note:** Your changes may require re-recording VCR cassettes for the tests, see [VCR Cassette Management](#-vcr-cassette-management).

4. **💾 Commit your changes** using Conventional Commits format:

   We use [Conventional Commits](https://www.conventionalcommits.org/) format for our commit messages. Instead of using regular git commit, please use commitizen:

   ```bash
   uv run cz commit
   ```

   This will guide you through an interactive prompt to create a properly formatted commit message with:
   - Type of change (feat, fix, docs, style, refactor, etc.)
   - Optional scope (e.g., api, cli, docs)
   - Short description
   - Optional longer description and breaking change notes

   Example of resulting commit message:

   ```text
   docs(readme): update installation instructions
   ```

   > **Note:** If pre-commit hooks fail or modify files during `cz commit`, you can retry with the same message:
   >
   > ```bash
   > uv run cz commit --retry
   > ```

## 🔄 Pull Request Process

1. **🔄 Update your branch** with the latest changes from the `dev` branch:

   ```bash
   git fetch upstream
   git checkout feature/your-feature-name
   git merge upstream/dev
   ```

2. **📤 Push your changes** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

3. **🎯 Create a pull request** from your branch to the `dev` branch. We use the `dev` branch for integration and testing before merging to `main` to keep the main branch stable for releases.

4. **📋 Fill out the pull request template** with all required information.

5. **✍️ Sign the Contributor Agreement** by including your filled-in `.github/contributors/[your-github-username].md` file (required in the first pull request).

6. **⏳ Wait for review**. Maintainers will review your PR and may request changes.

7. **🔧 Address review comments** if requested.

## 🐛 Issues and Feature Requests

When submitting issues or feature requests, please use our GitHub issue templates:

1. Check if a similar issue already exists in the [Issues](https://github.com/shcherbak-ai/contextgem/issues) section.

2. If not, create a new issue using the appropriate template:
   - **Bug Report**: For reporting bugs or unexpected behavior
   - **Feature Request**: For suggesting new features or enhancements
   - **Documentation Improvement**: For suggesting improvements to our documentation

Each template will guide you through providing all the necessary information for your specific request.

Do not report a security vulnerability in a public issue: report it privately, as described in [SECURITY.md](SECURITY.md).

By submitting issues or feature requests to this project, you acknowledge that these suggestions may be implemented by the project maintainers without attribution or compensation.

## 🧪 Testing

### 🏗️ Current Test Structure

Tests are split across the feature/integration test modules, two files for pure helpers and the tests of the changelog:

- **`tests/test_<area>.py`** (one `Test<Area>` class per module) — feature and integration tests, including all LLM-backed extraction tests that replay through VCR cassettes. Most contributions belong here, in the module of the relevant feature area:

  | Module | Feature area |
  |--------|--------------|
  | `test_aspects_and_concepts.py` | Aspects, concepts, examples and extracted items; extraction of concepts of specific types |
  | `test_chat.py` | Messages, chat sessions, chat with LLMs and tool calling |
  | `test_decision_models.py` | Decision models: questions for concepts and aspects, conversion of answers to item values and probabilities; extraction of aspects and concepts with decision models, on their own, in model groups and in fallback chains |
  | `test_docling_properties.py` | Property-based tests of the creation of documents from Docling documents: generated Docling documents of the structures that Docling gives, and the paragraphs, contexts, pages and images that each must give |
  | `test_documents.py` | Documents, paragraphs, sentences and extraction pipelines; text segmentation with SaT models |
  | `test_extraction.py` | Extraction of aspects and concepts from documents; extraction errors |
  | `test_extraction_properties.py` | Property-based tests of the extraction with LLMs and decision models, whose APIs are stubbed: the items, errors and usage that any output of the models entails; retries, fallback models, model groups and concurrency |
  | `test_images.py` | Images; extraction from images and multimodal documents |
  | `test_internals.py` | Prompt templates, registry of public types, base classes; rejection of invalid names and assignments |
  | `test_llms.py` | LLM configuration, local LLMs, prompts, token limits, retries of API requests and rate limiting |
  | `test_logging.py` | Logger settings and log levels |
  | `test_model_groups.py` | Model groups, model roles and fallback models |
  | `test_performance.py` | Extraction benchmarks |
  | `test_serialization.py` | Serialization, deserialization and cloning of instances |
  | `test_usage_and_cost.py` | Usage and cost tracking, model pricing |
  | `test_usage_examples.py` | Usage examples in the documentation, README and docstrings |

  All these test classes inherit from `BaseTest` in **`tests/base.py`**, which creates the test objects (documents, LLMs, images, etc.) once and shares them across the test classes. A test must leave the shared objects unchanged: a test that changes a document works on its copy (`document = self.document.clone()`). The usage and cost recorded by the shared LLMs are reset before each test. The tests do not depend on the order in which they are run.
- **`tests/test_properties.py`** and **`tests/test_units.py`** — property-based and example-based tests of pure utility helpers (see [below](#-tests-for-pure-helpers-property-based-and-example-based)).
- **`tests/test_changelog.py`** — checks that each version in `CHANGELOG.md` has its compare link, and that `CITATION.cff` gives the version of the package and the release date of that version in the changelog.

### 📚 Tests of the Usage Examples

Each usage example in `dev/usage_examples/docs/` and `dev/usage_examples/readme/` is run by a test of its own in `tests/test_usage_examples.py`. The examples are listed there by the names of their modules, in one of two lists:

- `_EXAMPLES_WITH_REQUESTS` — the examples that send requests to the API of a model. Each has a cassette of its own (`TestUsageExamples.test_usage_example[<example>].yaml`), so when the requests of an example change, delete and re-record the cassette of that example only:

  ```bash
  uv run pytest "tests/test_usage_examples.py::TestUsageExamples::test_usage_example[docs.quickstart.quickstart_aspect]"
  ```

- `_EXAMPLES_WITHOUT_REQUESTS` — the examples that send no requests, e.g. the ones that only create objects. They are run with all network access blocked.

When you add an example, add it to one of the lists: a test fails for an example that is in neither. The examples of the docstrings (`dev/usage_examples/docstrings/`) are run by `test_docstring_examples`.

### 📏 Testing Guidelines

- Write tests for new features or bug fixes
- Make sure all tests pass before submitting a PR. For a substantial change, run them on every supported Python version, as CI does: `uv run --isolated --python 3.11 pytest` runs them on Python 3.11 and leaves the environment of the project as it is
- Maintain code coverage above **80%**
- Check code coverage by running:

  ```bash
  uv run pytest --cov=contextgem
  ```

---

### 🎲 Tests for Pure Helpers (Property-Based and Example-Based)

Pure utility helpers in `contextgem/internal/` (text cleaning, JSON parsing, type-hint serialization, list chunking, tool-schema generation, etc.) are covered by two complementary test files. Both run without LLMs, shared test objects, or cassettes.

**`tests/test_properties.py`** — property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/). Property tests assert **invariants that must hold for any input** ("running the function twice yields the same result", "concatenating chunks recovers the original list", "serialize then deserialize round-trips") rather than checking hand-picked input/output pairs. Hypothesis generates many varied inputs and shrinks any failure to a minimal reproducer.

Reach for a property test when:

- You're working with a pure deterministic function with an abstract contract.
- You can describe correctness as one or more invariants (idempotence, round-trip, bounds, length-preservation, etc.).

**`tests/test_units.py`** — example-based unit tests that pin specific input/output contracts. Reach for a unit test when:

- The function's correctness is "this specific output for this specific input" (e.g., "this Python type maps to this JSON schema").
- You need to assert a specific error message or exception type.
- The behavior is contract-shaped rather than invariant-shaped.

Stick with the feature/integration test modules (`tests/test_<area>.py`) for feature/integration scenarios where correctness depends on full document/aspect/concept extraction flows — most LLM-backed tests fall into this category.

---

### 🧬 Property-Based Tests of the Extraction

**`tests/test_extraction_properties.py`** tests the extraction itself with [Hypothesis](https://hypothesis.readthedocs.io/), with LLMs and with decision models, whose APIs are stubbed: no requests are sent, and no cassettes are needed. Unlike the tests for pure helpers, its tests use the shared test objects of `BaseTest` and run whole extractions. Hypothesis generates the documents, the aspects and concepts, and what the APIs return, including what a model is not supposed to return: IDs that are not in the prompt, values of another type than the concept has, numbers that are not valid JSON, probabilities that are not numbers within [0, 1], failed requests, and responses cut off at the max output tokens. Most of its tests check the outcome of the extraction against what these responses entail: the items extracted, or the error or the warning, and the calls and tokens recorded; the test of fallback chains also checks the cost. Others check invariants: a higher decision threshold never adds items, and any configuration of aspects and concepts round-trips through serialization. The tests run the retries for invalid data, the fallback models, the model groups and concurrency. Only hosted models are used, as where a model is served does not change how its responses are handled.

When you change how the output of a model is read, how the calls are retried or passed to fallback models, or how usage is recorded, change the oracle of the test concerned with it (the `_get_expected_*` functions and `_simulate_extraction`), so that the tests state the new behavior. A failure is reported with the generated document, the response and the expected outcome, shrunk to a minimal example.

---

### 📊 Extraction Evaluation

The tests check what the extraction does; **`dev/evals/`** measures how well it does it, and at what cost. It extracts evaluation cases, i.e. complex documents with reference answers (two contracts, the technical documentation of a microcontroller and the stability study report of a tablet), with real models, and scores the aspects, the concepts and their references against the answers. A decision model used as a judge (`typesafe/jev-latest` by default) scores what cannot be compared exactly: narrative answers, justifications and text fields. The judge answers one yes/no question at a time with a probability, and is checked first on answers whose verdict is known. Recorded baselines give the scores and the costs of the models at one state of the library, and `uv run fab costs` adds up the costs that the results record. The harness is not a part of the package, and the tests do not run it.

A change that can affect the quality of the extraction, e.g. to a prompt template, to how the output of a model is read, or to a default, is measured there: the version with the change and the version without it are run at the same time and compared. The harness sends live requests, which are paid for and need the API keys of the providers, so the maintainer runs it; say in your pull request if you think a change needs it. Its rules, commands and baselines are in [dev/evals/AGENTS.md](dev/evals/AGENTS.md).

### 📄 Docling Fixtures

The tests of `Document.from_docling()` use files in each format that Docling converts, in `tests/other_files/docling/`, each with the Docling document that full Docling converts it into (`<file name>.json`): files written for ContextGem, and in `docling_tests/`, test documents of Docling, copied from its repository under its MIT License (the `LICENSE` file there, and `NOTICE`). Add only files that may be redistributed: written for the tests, or under a license that allows it, never e.g. papers, articles or web pages of others. The tests convert the files in the formats that need no models themselves (`TEST_CONVERTED_SUFFIXES` in `dev/docling_fixtures.py`: DOCX, XLSX, PPTX, HTML, Markdown, CSV, AsciiDoc, LaTeX, WebVTT, EPUB, JATS and USPTO XML), with the torch-free `docling-slim` of the dev group, and check that the result matches the committed Docling document. A PDF file needs the layout and table models of Docling, which run on PyTorch, and an email (EML, MSG) or OpenDocument file (ODT, ODS, ODP) packages that the dev group does not have: the tests read their committed Docling documents only.

Next to each file, `<file name>.contextgem.jsonl` holds what ContextGem makes of its committed Docling document: a line for each paragraph, with its text, additional context and page numbers, and a line for each image. `test_from_docling_fixture_outputs` checks that ContextGem still makes it, so that a change of the converter, or of the Docling documents after an upgrade of Docling, shows in review as a change of these outputs. When the new output is right, write the outputs with `uv run fab docling-fixtures --outputs`, which reads the committed Docling documents without Docling, and commit them with the change.

`tests/test_docling_properties.py` generates Docling documents with [Hypothesis](https://hypothesis.readthedocs.io/), in the structures that Docling gives the files it converts (headings, also with formatted text, nested lists, formatted text with pictures in it, tables, pictures, key-value regions, form fields, sheets and slides, and the furniture and the notes of the pages), with a word of its own in each text. One test works out from the structure the paragraphs that the document must have, with their texts, contexts and page numbers, and its images (`_Expected`); another checks what holds for any Docling document, e.g. that each text of the body is kept and that the furniture is left out. When you change how the converter converts a structure, change `_Expected` with it, so that the test states the new behavior.

`uv run fab docling-fixtures` converts all the files with full Docling, the version in `uv.lock`, and the OpenDocument support of `docling-slim`, which full Docling does not install, in a throwaway environment with the CPU-only PyTorch, and checks that ContextGem makes the same paragraphs from them as from the committed Docling documents; `--write` writes the Docling documents and then the outputs of ContextGem from them instead, and `--docling-version latest` (or `X.Y.Z`) uses another Docling. It downloads the models of Docling on the first run (about 0.5 GB) and sends no request to a model. On Windows, the page renderer of docling-parse crashes the conversions now and then ([docling-parse#357](https://github.com/docling-project/docling-parse/issues/357)), so the script runs them again, up to ten times. The workflow `.github/workflows/docling-fixtures.yml` runs the check when `uv.lock` or the converter changes, and weekly with the latest Docling. When an upgrade of Docling changes the output, check the difference, adapt the converter if the new output needs it, and otherwise write the Docling documents again and commit them.

The tests run with the docling-core of `uv.lock`. `uv run fab docling-floor` runs the tests of `Document.from_docling()` with the oldest one that the `docling` extra allows: in a throwaway environment created from `uv.lock`, it replaces docling-core with its lower bound and uninstalls docling-slim, which needs a newer one, as for a Docling document converted elsewhere, and checks the environment with `uv pip check`. The test that converts the fixture files with docling-slim, `test_from_docling_converted_files`, is left out; the others read the committed Docling documents. It takes a minute or two and sends no request to a model, and CI runs it on each push. When it fails, the converter needs a newer docling-core: raise the lower bound in `pyproject.toml` (the extra and the dev group) and `_MIN_DOCLING_CORE_VERSION` together.

### 📦 Install Check of a Release

The tests run on the locked dependencies, on the platform they run on; a new install gets the newest dependencies that the requirements admit, on its own platform. `uv run fab install-check` resolves the dependencies of `pyproject.toml`, without and with its extras, for Linux (glibc and musl), Windows and macOS, on x86_64 and arm64, and for each supported Python, from wheels only, as an install without a compiler gets them. uv answers for any platform from the wheels that PyPI lists, so the check takes seconds, needs no machine of the platform, and sends no request. Run it when you change the version or the range of a dependency; the maintainer runs it before each release. `--version latest` (or `--version X.Y.Z`) checks a release as PyPI serves it. The installs known to fail are listed with their reasons in [dev/check_install.py](dev/check_install.py), and the check fails on any difference from the list: a new failure, or an install that no longer fails. Then update the list, and the documentation that names the gap, e.g. the note on Alpine in `docs/source/installation.rst`.

---

### 📼 VCR Cassette Management

We use [pytest-recording](https://github.com/kiwicom/pytest-recording) to record and replay HTTP interactions with LLM APIs (both cloud-based and local). This allows tests that call LLM APIs to run without making actual API calls after the initial recording.

#### Why VCR Cassettes?

VCR cassettes provide the most reliable testing approach for ContextGem because:

- **Real API Testing**: Testing with actual LLM APIs ensures our functionality works as expected with real responses, edge cases, and API behaviors
- **Scalability**: With a significant number of LLM API tests, hardcoding requests/responses would be impractical and unmaintainable
- **Reproducibility**: Once recorded, tests run consistently without variability in LLM responses
- **No Setup Friction**: Contributors can run tests without API keys or local LLM installations

Local LLMs (Ollama, LM Studio, etc.) also use HTTP APIs (typically on localhost) and their interactions are recorded in cassettes too.

The test suite automatically uses dummy environment variables with pre-recorded cassettes when there is no `.env` file, or one that sets no variable, so most contributors won't need to set up real API keys or local LLM servers. The first test run downloads the SaT segmentation model of the tests and its tokenizer from HuggingFace (several hundred megabytes in all), so it needs network access.

#### Determining Your Scenario

To determine whether you need to record new or re-record existing cassettes, **run the tests first**:

```bash
uv run pytest
```

Based on the test results and your changes, you'll fall into one of these four scenarios:

---

#### ✅ Scenario 1: No Cassette Recording Required

**When this applies:**

- New tests that **do not** call LLM APIs
- Code changes that don't modify internal prompts or LLM parameters
- Changes are compatible with existing pre-recorded API calls (confirmed by passing tests)

**What to do:**

- Nothing! Tests that call LLM APIs should pass by replaying from existing cassettes with automatically-set dummy environment variables
- No need to create a `.env` file or set up API keys

---

#### 🆕 Scenario 2: New Cassettes Need Recording

**When this applies:**

- New test methods that call LLM APIs (cloud-based or local)
- Adding tests for new functionality that requires LLM interaction

**What to do:**

1. **Create a `.env` file** locally (ignored by git) from the template `.env.example`, then uncomment and fill in the variables of the APIs that your new tests call. The template says which tests need each variable:

   ```bash
   cp .env.example .env
   ```

   Every variable in `.env.example` is commented out, so a copy that is left as it is sets none, and the tests keep using the dummy variables. Once your `.env` file sets a variable, the tests set no dummy ones: `CONTEXTGEM_OPENAI_API_KEY` and `CONTEXTGEM_TYPESAFE_API_KEY` then need a value, any value if you only replay their tests, as `litellm` rejects a request without a key before it can be replayed from a cassette. The other variables fall back to their dummy values when they are not set.

2. **For new LLM providers**, create environment variables prefixed with `CONTEXTGEM_`:

   ```bash
   CONTEXTGEM_GOOGLE_AI_STUDIO_API_KEY=your_google_api_key
   ```

   Also add the API host of the provider to `_TETHERED_RECORDING_ALLOW` in `tests/conftest.py`, otherwise its requests are blocked while recording (see [Network Egress Control](#network-egress-control)). If the provider does not send its key in the `Authorization` or `api-key` header (e.g. Google AI Studio sends it in `x-goog-api-key`), add that header to `VCR_FILTER_HEADERS` in `tests/utils.py`, otherwise the key is recorded in the cassette.

3. **Update dummy variables** by adding your new environment variables to `DUMMY_API_ENV_VARS` in `tests/utils.py`, mapped to a dummy value (e.g. "DUMMY"), and to `.env.example`, commented out, with a line on the tests that need them. A test checks that the two list the same variables.

4. **Add the VCR decorator** to your new test methods that call LLM APIs (cloud or local):

   ```python
   @pytest.mark.vcr
   def test_your_new_llm_feature(self):
       # Your test code that calls LLM APIs (cloud or local)
   ```

   > ⚠️ **Important:** Without the `@pytest.mark.vcr` decorator, no cassette will be recorded!

5. **Run your new tests** - new cassettes will be created automatically

6. **Verify redaction** - check that sensitive data is properly redacted in the new cassette files (see [Environment Variable Security](#environment-variable-security) for what is redacted)

7. **Test with dummy variables** - delete your `.env` file and run tests again to confirm LLM API tests pass by replaying from cassettes with dummy variables

---

#### 🔄 Scenario 3: Some Existing Cassettes Need Re-recording

**When this applies:**

- Tests fail because your changes are incompatible with specific existing cassettes
- Only certain test cases are affected

**What to do:**

1. **Identify failing cassettes** from test output

2. **Delete specific cassette files** from `tests/cassettes/` that need re-recording

3. **Create a `.env` file** if needed (same as Scenario 2)

4. **Run the affected tests** to re-record only the necessary cassettes:

   ```bash
   uv run pytest tests/test_extraction.py::TestExtraction::test_specific_method
   ```

---

#### 🔄🔄 Scenario 4: All Cassettes Need Re-recording

**When this applies:**

- You modified internal prompts (direct changes or code that renders prompts differently)
- You changed default LLM API parameters
- Multiple LLM-related tests fail due to your changes

**What to do:**

1. **Delete all cassette files**:

   ```bash
   # On Unix/Linux/Mac
   rm tests/cassettes/*.yaml

   # On Windows
   del tests\cassettes\*.yaml
   ```

2. **Create a `.env` file** with the API keys of all the services (same as Scenario 2), and set up what the tests of local LLMs and of decision models need (see [Local LLM Testing](#local-llm-testing) and [Decision Model Testing](#decision-model-testing))

3. **Run all tests** to re-record everything:

   ```bash
   uv run pytest
   ```

> ⚠️ **Important:** This will use significant API quota and may incur substantial costs! Before deleting the cassettes, run `uv run fab costs` to see what recording them again costs (see [Important Notes](#important-notes)).

---

#### Environment Variable Security

**What is redacted automatically:**

- `CONTEXTGEM_OPENAI_API_KEY`, `CONTEXTGEM_ANTHROPIC_API_KEY`, `CONTEXTGEM_TYPESAFE_API_KEY` and `CONTEXTGEM_AZURE_AI_API_KEY`: the headers that carry these keys (`Authorization`, `x-api-key`, `api-key`) are among the headers listed in `VCR_FILTER_HEADERS` (`tests/utils.py`), whose values are redacted in the recorded requests and responses. A key that is sent in another header, in the URL or in the body of a request is not redacted.
- `CONTEXTGEM_AZURE_AI_API_BASE`: the host of a Microsoft Foundry resource, which has the name of the resource, is replaced with a dummy host (`dummy-endpoint.local`) in the recorded requests, and the name of a project of the resource with a dummy name, if the endpoint has it

The response headers in which Azure names the deployment, the region and the cluster that served a request are redacted too, as are those in which Anthropic names the organization and the workspace of the key, and the rate limits of the account. The name of your home directory is also redacted in the file paths found in the recorded responses, e.g. in the paths of the model files that a local LLM server returns.

#### Local LLM Testing

For local LLM testing, install the following tools and download the relevant models identified under `ollama` and `lm_studio` prefixes in `tests/test_llms.py`:

- [Ollama](https://ollama.com/)
- [LM Studio](https://lmstudio.ai/)

> ⚠️ **Important:** Your system needs to have an appropriate GPU capacity to run such local LLMs.

#### Anthropic LLM Testing

Most tests of LLMs send their requests to OpenAI. The tests `test_anthropic_llms_*` in `tests/test_llms.py` send theirs to Anthropic (`anthropic/claude-haiku-5-5` and `anthropic/claude-sonnet-5-5`): recording their cassettes needs `CONTEXTGEM_ANTHROPIC_API_KEY` in your `.env` file. A dummy `CONTEXTGEM_ANTHROPIC_API_KEY` is set when no `.env` file sets a variable, and the tests use the same dummy key if your `.env` file has no key of Anthropic, so these tests replay without the variable. The host of the API of Anthropic (`api.anthropic.com`) is on the list of the hosts allowed while recording (see [Network Egress Control](#network-egress-control)).

#### Decision Model Testing

Most tests of decision models in `tests/test_decision_models.py` replace the API call of `litellm` with a stub and need no cassettes. The tests that send requests to the APIs of decision models are marked with `@pytest.mark.vcr`: the ones in `tests/test_decision_models.py`, and the tests of the usage examples in `dev/usage_examples/docs/decision_models/` (`test_usage_example[docs.decision_models.<example>]` in `tests/test_usage_examples.py`). Recording their cassettes needs:

- **`CONTEXTGEM_TYPESAFE_API_KEY`** in your `.env` file, for the requests to TypeSafe (`typesafe/jev-latest`).
- **`CONTEXTGEM_OPENAI_API_KEY`** in your `.env` file, for the requests to the Decisions API of OpenAI (`openai/gpt-6-luna` as a decision model), and to the LLM that is used next to the decision model in a model group.
- **`CONTEXTGEM_AZURE_AI_API_KEY` and `CONTEXTGEM_AZURE_AI_API_BASE`** in your `.env` file, for the requests to a decision model deployed in Microsoft Foundry: the key and the endpoint of your Foundry resource (`https://<resource>.services.ai.azure.com`). The resource needs a deployment of Microsoft-Decision-1 named `ms-decision-model`: a model deployed in Microsoft Foundry is requested by the name of its deployment, which is the one in `tests/test_decision_models.py` (`azure_ai/ms-decision-model`). It must have no deployment named `Microsoft-Decision-1`, which one of the tests requests to get the error for a deployment that does not exist.
- **[Ollama](https://ollama.com/) 0.35 or later**, running on its default port, with the decision model identified under the `strands_decider/` prefix in `tests/test_decision_models.py` downloaded (`ollama pull nimble`), for the requests to a local decision model. No API key is needed.
- **[LM Studio](https://lmstudio.ai/) 0.4.26 or later** (not LM Studio Bionic, whose server does not serve decision models), running its local server on the default port, with the decision model of the LM Studio test in `tests/test_decision_models.py` downloaded (`lms get https://huggingface.co/ggml-org/Kev-4B-GGUF`). No API key is needed.

`litellm` rejects a request to a decision model that has no API key before the request is sent, i.e. before it can be replayed from a cassette. For this reason, a dummy `CONTEXTGEM_TYPESAFE_API_KEY` is set when no `.env` file sets a variable. If your `.env` file has no `CONTEXTGEM_TYPESAFE_API_KEY`, add the variable with any value, otherwise these tests fail also in replay.

A decision model deployed in Microsoft Foundry is requested with a key and with the endpoint of the resource. A dummy `CONTEXTGEM_AZURE_AI_API_KEY` and a dummy `CONTEXTGEM_AZURE_AI_API_BASE` (`https://dummy-endpoint.local`, the host that the host of a resource is replaced with in the cassettes) are set when no `.env` file sets a variable. The tests use the same dummy values if your `.env` file has no key or endpoint of a Foundry resource, so these tests replay without the two variables.

The host of TypeSafe (`api.typesafe.ai`) is on the list of the hosts allowed while recording. The host of your Microsoft Foundry resource is allowed while recording as well: it differs for each resource, so it is read from `CONTEXTGEM_AZURE_AI_API_BASE`, and is accepted only if the endpoint is an HTTPS one of a host under `services.ai.azure.com`. A test that sends requests to another provider of decision models needs the host of that provider added to that list (see [Network Egress Control](#network-egress-control)).

#### Important Notes

> **💰 Cost Warning:** Recording cassettes for test methods that use live LLM API (non-local LLMs) uses your API keys and **will incur charges**. Scenario 4 (re-recording all cassettes) can be particularly expensive.
>
> **💵 What recording costs:** Each recorded response keeps the usage of its request, so the cost of the cassettes is computed from them, without sending a request. `uv run fab costs` prices the requests in all the cassettes at the current prices of `genai-prices`, i.e. what recording them again would cost, by model and for the most expensive cassettes. `uv run fab costs --changed` prices only the cassettes that differ from the last commit. A test run reports at its end, after the results of the tests, what the requests of the cassettes it recorded cost, by model, and what those of the cassettes it replayed would cost if they were sent live. Requests to local models are free. A model that `genai-prices` has no price for gets its price in `tests/cassette_costs.py`: the report names the models that have none.
>
> **📼 Cassette names:** Cassette files are named `<TestClass>.<test name>.yaml`. If you move a test that calls LLM APIs to another test class, rename its cassette files accordingly (with `git mv`) in the same change. Otherwise, the test will not find its cassette and will record a new one.

#### Network Egress Control

The test suite uses [tethered](https://github.com/shcherbak-ai/tethered), a development dependency, for network egress control: the `_tethered_network_guard` fixture in `tests/conftest.py` enforces it at the socket level during every test:

- **Tests without `@pytest.mark.vcr`**: all outbound connections are blocked, including the ones to localhost
- **VCR-marked tests, replay mode** (cassette exists): blocks all outbound connections except HuggingFace (for SaT model downloads not captured by VCR)
- **VCR-marked tests, recording mode** (no cassette): allows only approved endpoints (LLM APIs, HuggingFace for model downloads, genai-prices for cost data), the Microsoft Foundry resource whose endpoint is set in `CONTEXTGEM_AZURE_AI_API_BASE`, and localhost for local LLMs

The fixture activates tethered in **hardened mode** (`locked=True` with a private lock token, `external_subprocess_policy="block"`), so test code can't disable the policy mid-test and non-Python subprocesses are refused — closing the shell-egress bypass.

The policy above applies while a test runs. Before the tests start, the `_warm_token_encodings` session fixture counts tokens once, so that the token encodings used by `litellm` are available during the tests. On Windows, this downloads the encodings if `tiktoken` rejects the bundled ones (see the Windows note in [Running Tests](#-running-tests)). The download is wrapped with `tethered.scope` that allows a single host (`openaipublic.blob.core.windows.net`), which is deliberately not a part of the replay and recording allowlists, and `tiktoken` verifies each downloaded encoding against the hash pinned in its package.

If you add tests that connect to new endpoints, update the `_TETHERED_RECORDING_ALLOW` list in `tests/conftest.py`.

---

### 🏃 Running Tests

Run all tests:

```bash
uv run pytest
```

Run specific tests:

```bash
# Run a specific test method
uv run pytest tests/test_extraction.py::TestExtraction::test_extract_all
```

**🪟 Windows only**: The first test run after `litellm` is installed or reinstalled in your virtual environment (e.g. after upgrading the dependencies) needs network access, to download the token encodings before the tests start. This happens automatically. If the download fails (e.g. you are offline or behind a proxy), a warning is logged at the start of the test run, and several unrelated-looking tests fail, such as the local LLM tests, `test_input_output_token_validation`, and `test_very_long_doc_extraction`. In that case, run the following command once, with network access, and run the tests again:

```bash
uv run python -c "import litellm; litellm.token_counter(model='gpt-6-luna', text='test')"
```

> **Why:** The Windows build of `litellm` bundles the token encoding files with Windows line endings, which the tokenizer library (`tiktoken`) rejects, so that the encodings are downloaded on the first token count instead. During a test, this download is blocked (by VCR in replay mode, by `tethered` in the tests that are not VCR-marked), in which case tokens cannot be counted. Therefore, the encodings are downloaded before the tests start (see [Network Egress Control](#network-egress-control)) and cached in the virtual environment. The Linux and macOS builds of `litellm` are not affected.

### ⚠️✅ Expected Test Warnings

Warnings generated during tests are often expected and by design. Many warnings are intentionally triggered to test error handling, edge cases, and warning systems. Common expected warnings include:

- Model extraction errors and retries (testing error handling)
- Deprecation warnings from dependencies

**Key Point**: If tests **PASS** with warnings, this should **not** prevent you from submitting your PR. The test suite is designed to handle and expect these warnings as part of normal operation.

### 🐞 Debugging Tests

The tests run with the ContextGem logger at the `DEBUG` level, which is set in `tests/conftest.py`, so the log output shows detailed information about test execution.

## 📚 Documentation

- Update the docstrings and the pages in `docs/source/` that describe the functionality you change or add, in the same pull request. Use clear, concise language.
- The code shown on a page or in a docstring lives in `dev/usage_examples/` and is included with `literalinclude`, so that the tests run it (see [Tests of the Usage Examples](#-tests-of-the-usage-examples)). Add or change the example there, not in the page.
- Update `dev/readme.template.md` and `PYPI.md` where the change affects them (see [Updating README.md](#-updating-readmemd)).

### 🏗️ Building the Documentation

Use the fabric commands from the project root:

#### For Live Development (Recommended)

```bash
# Live rebuild with auto-refresh on file changes
uv run fab docs-live
```

This starts a development server on `http://127.0.0.1:9000` with:

- Automatic rebuilds when files change
- Browser auto-refresh
- Pretty URLs without `.html` extensions

#### For Static Builds

```bash
# Build with verbose output, ignore cache, and treat warnings as errors
uv run fab docs
```

### 👀 Viewing the Documentation

**With Live Development:** open `http://127.0.0.1:9000` in your browser.

**With Static Builds:** the pages of a static build link to each other as directories, which a browser does not resolve when the files are opened from disk. After building, serve the folder and open `http://127.0.0.1:9000`:

```bash
uv run python -m http.server 9000 --bind 127.0.0.1 --directory docs/build/dirhtml
```

Use `127.0.0.1` rather than `localhost`: on Windows, a request to `localhost` tries IPv6 first and waits two seconds for the refusal of a server that listens on IPv4 only.

### 🌐 Live Documentation

You can access the live documentation at: <https://contextgem.shcherbak.ai>

> **Note:** Documentation is automatically deployed when maintainers merge changes from `dev` to `main`. As a contributor, your documentation changes will be visible on the live site after your PR is merged and subsequently deployed by maintainers.

### 📋 Documentation Structure

Under `docs/`:

- `source/` - Contains the source `.rst` files. Each page starts with a `.. meta::` block whose `:description:` says in one sentence what the page covers: it is the page's line in `llms.txt`, the index of the documentation for LLMs and coding agents that the build writes to the root of the site, next to a Markdown copy of each page
- `source/_static/` - Static assets like images
- `source/conf.py` - Sphinx configuration
- `build/` - Generated documentation (not committed to version control)

### 📝 Updating README.md

The project's README.md is generated from a template `dev/readme.template.md`, as it embeds code fragments that are located in separate modules and are subject to tests. Do not modify README.md directly as your changes will be overwritten by a pre-commit hook.

Instead:

1. Edit the template file at `dev/readme.template.md`
2. The pre-commit hook will automatically update README.md using the template

If you need to test the README generation manually:

```bash
uv run fab readme
```

**Links and images.** The README is shown on GitHub. In the template, link to the documentation site for anything it documents, and use relative paths for the files of the repository. Images come from `docs/source/_static/` (shared with the documentation site) or `assets/` (the README's own), by relative paths, so that they render in a local preview of `README.md` and on any branch:

```markdown
[Contributing](CONTRIBUTING.md)
![Description of the image](docs/source/_static/example.svg)
```

Give images descriptive alt text, and check them at their README display size on light and dark backgrounds. Label fictional examples as illustrative.

- `docs/source/_static/contextgem_wordmark.svg` is pixel art, 144 by 18 pixels, in one palette for both GitHub themes. Show it at a whole multiple of that size (`width="576"`), so that its pixels stay square. Its gem catches the light once, when the image is loaded.
- `assets/licenseal-wordmark.svg` and `assets/tethered-wordmark.svg` are the sibling projects' pixel names, drawn at twice their size in transparent images of one size (336 by 80), so that the two share a baseline. Show them at that width.
- `assets/contextgem-separator.svg` is the pixel rule between the README's main parts: the wordmark's gem, at rest, in a full context meter, as a terminal draws one: the whole document is in context, and the gem is what comes out of it. On either side of the gem is a bar of full blocks, three pixels high, lit along its top, shaded at its foot and over a darker copy of itself, as the wordmark's strokes are, with a seam every fourth pixel counted from the gem. The bars are in a group at 60% opacity, so that they recede on both GitHub themes and the gem, at full strength, is what the eye finds. It is pixel art of 228 by 13 pixels, one path of rectangles per colour for the bars and for the gem. Show it at twice that size (`width="456"`) with an empty `alt`, and keep section headings as text, so that their anchors, GitHub's outline and page search keep working.
- `assets/icons/` holds the README's icons: pixel art of 16 by 16 pixels, one path of rectangles per colour. Show an icon at twice that size (`width="32" height="32"`) with an empty `alt`, above the title of a card or of a table column, and never in a heading, where it changes the heading's anchor. Lines are in the gem's mid blue (`#0e74bc`), which reads on both GitHub themes, and the part that carries the meaning is in its light blue (`#38bdf8`). The palest blue (`#a5ebff`) and white disappear on a light ground: use them only for a single pixel of shine. An icon stands for one thing wherever it appears, such as `llm.svg` and `decision-model.svg` for the two kinds of models. The `format-*.svg` icons stand for file formats that Docling converts, in a table of four columns under the Docling quick start, which fits the width of a phone; the table names the formats, and links to Docling's list of all of them.
- The figures (`contextgem_readme_extraction.svg`, `contextgem_readme_models.svg`, `contextgem_readme_aspects_concepts.svg`, `contextgem_readme_decision_model.svg`, `contextgem_docling.svg`, `contextgem_readme_structure.svg`, `contextgem_agentic_workflows.svg`) are drawn on their own dark ground, so one file serves both GitHub themes. Draw a new one in their style: hairlines, monospace labels, grey, and the blue of the gem only for what is extracted.
- Do not delete an image that a released version of the README loads from `https://contextgem.dev/_static/`.
- `social/github-social-preview.svg` is the picture a link to the repository, or to a page of the documentation, is shown with on other sites: a pixel-art scene, five pixels to a pixel of the art, with a card that carries the wordmark at rest, the README's heading, the tagline's claims and the company's name. When one of these changes, change the SVG with it and render `social/github-social-preview.png` with `uv run fab social-preview` (the task needs Chrome, Edge or Chromium). GitHub reads no file for it: a maintainer uploads the picture in the settings of the repository. In the SVG a rule in `<style>` outranks an attribute: set type sizes by class, and write no rule for `path`, which would override the art's fills.
- `docs/source/_static/favicon.ico` (16 to 256 pixels) and `docs/source/_static/contextgem_gem.svg`, the logo of the documentation's sidebar, are the wordmark's gem at rest. Change them with it.

Look at a change as GitHub will show it before pushing it. `uv run fab readme-preview` serves `README.md` in GitHub's light theme at <http://127.0.0.1:9120/> and in its dark theme at <http://127.0.0.1:9121/> (`--port N` serves them at N and N+1). It shows the file as it is, so regenerate it from the template first (`uv run fab readme`):

- The HTML is what GitHub's Markdown API returns for the file, so what GitHub removes from a README is removed here too. The API returns an alert (`> [!NOTE]`) as a quote in the mode that gives the headings their anchors, so the preview takes the alerts from a second request in its other mode, and shows them as the repository's page does.
- The look comes from github-markdown-css, in GitHub's colours, each page in its theme whatever the browser is set to. The column is as wide as GitHub's at 1280 pixels and wider, and below 768.
- A reload sends the README to GitHub again when it has changed, once for both pages, twice where it has alerts. GitHub allows 60 requests an hour without a token.

**Package description.** PyPI shows [PYPI.md](PYPI.md) (`project.readme` in `pyproject.toml`), not the README. Keep it short, without images, with absolute HTTPS links, and update it when the summary, the install command or the feature list changes.

## ❓ Questions & Support

We're here to help! Whether you're stuck on something technical, have questions about the contribution process, or want to suggest improvements to this guide, don't hesitate to reach out.

### 🆘 Get Help With

- **Technical Issues**: Setup problems, test failures, or development environment issues
- **Contribution Process**: Questions about pull requests, branching, or code review
- **Feature Ideas**: Discussion about new features or improvements
- **Documentation**: Clarifications about this contributing guide or suggesting improvements

### 📞 Contact Options

**💭 GitHub Discussions** (preferred for questions):

- [Start a discussion](https://github.com/shcherbak-ai/contextgem/discussions/new/)

**🐛 GitHub Issues** (for bugs, feature requests and documentation improvements):

- [Open a new issue](https://github.com/shcherbak-ai/contextgem/issues/new) using our issue templates

**📧 Direct Contact**:

- **📧 Email**: <sergii@shcherbak.ai>
- **💼 LinkedIn**: [Sergii Shcherbak](https://www.linkedin.com/in/sergii-shcherbak-10068866/)
- **🐦 X**: [@seshch](https://x.com/seshch)

### 📖 Improving This Guide

Found something unclear in this contributing guide? Missing information that would have helped you? Please:

- Open an issue using the Documentation Improvement template
- Suggest specific improvements or additions
- Share your contributor experience to help us improve the process

---

**Thank you for contributing to ContextGem!** 🙏

Your contributions help make ContextGem better for everyone. We appreciate your time and effort!
