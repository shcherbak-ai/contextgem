# 🤝 Contributing to ContextGem

Thank you for your interest in contributing to ContextGem! This document provides guidelines and instructions for contributing to the project.

## 📋 Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) to understand the expectations for all interactions within the project.

## ✍️ Contributor Agreement

Before your contribution can be accepted, you must sign our [Contributor Agreement](/.github/CONTRIBUTOR_AGREEMENT.md). This is a legal document that grants us the necessary rights to use your contribution. The agreement is based on the [Oracle Contributor Agreement](http://www.oracle.com/technetwork/oca-405177.pdf) and this requirement follows [OpenSSF Best Practices](https://www.bestpractices.dev/en) for silver-level criteria (section "Project oversight").

To sign the agreement:

1. Read the [Contributor Agreement](/.github/CONTRIBUTOR_AGREEMENT.md) carefully
2. Create a copy of the agreement under `.github/contributors/[your-github-username].md`
3. Fill in all the requested information and include it in your first pull request

## 🤖 Using AI Coding Assistants

This repository is **AI agent-friendly** and includes configuration files to help AI coding assistants understand the codebase:

- **[AGENTS.md](AGENTS.md)** - Project overview, architecture patterns, coding conventions, and workflow guidelines for AI assistants ([agents.md standard](https://agents.md))
- **[CLAUDE.md](CLAUDE.md)** - Configuration for [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)

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
    # Install uv if you don't have it
    pip install uv

    # Install dependencies and pre-commit hooks
    uv run fab setup

    # Or manually:
    # uv sync --all-groups --upgrade
    # uv run pre-commit install
    # uv run pre-commit install --hook-type commit-msg
    ```

### 🛠️ Available Fabric Commands

The project includes a `fabfile.py` with common development tasks:

```bash
uv run fab --list          # List all available commands
uv run fab setup           # Set up dev environment (deps + hooks)
uv run fab sync            # Sync dependencies with upgrades
uv run fab lint            # Run pre-commit checks on all files
uv run fab docs            # Build documentation
uv run fab docs-live       # Start live documentation server
uv run fab readme          # Regenerate README.md from template
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
├── pyproject.toml                # ⚙️ Dependencies & project config
└── README.md                     # 🤖 Auto-generated (don't edit)
```

**🎯 Quick Start for Your Contribution:**

- **Adding new functionality?** → Implement in `contextgem/internal/` (core logic). Then expose via a thin public facade in `contextgem/public/` using the registry.
- **Writing tests?** → Feature/integration tests go to the module of the relevant feature area (`tests/test_<area>.py`, e.g. `tests/test_extraction.py::TestExtraction`). Property tests for pure helpers go to `tests/test_properties.py`. Example-based unit tests for pure helpers go to `tests/test_units.py`.
- **Updating docs?** → Edit files in `docs/source/` or `dev/`
- **Fixing README?** → Edit `dev/readme.template.md`

> **💡 Note:** Implement functionality in `internal/` (base classes, validation, serialization, typing). Use `public/` to expose thin, documented facades that inherit from internal classes and are registered with `@_expose_in_registry` decorator to ensure deserialization and instance creation utils return public types. Do not import public classes in internal modules; use the registry for type resolution and publicization.

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

2. **📝 Make your changes** following our code style guidelines.

    We use several tools to maintain code quality:

    - **Ruff**: For code formatting and linting
    - **ty**: For static type checking
    - **Bandit**: For Python security vulnerability scanning
    - **Deptry**: For dependency health checks (unused, missing, transitive dependencies)
    - **Interrogate**: For docstring coverage checking
    - **markdownlint-cli2**: For Markdown style and consistency
    - **Twine**: For sdist/wheel metadata validation (via the `package-check` pre-commit hook, which runs `uv build` + `twine check`)
    - **uv lock check**: To verify that `uv.lock` is in sync with `pyproject.toml`
    - **Commitizen**: For the format of commit messages (see below)
    - **licenseal**: For license compatibility of the dependencies (in CI)
    - **Pre-commit hooks**: To automatically check and format code before commits

    The pre-commit hooks check and format your code when you commit. Some of them run with the Python of the project environment, so commit through uv (`uv run cz commit`, see below) or with `.venv` activated.

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

By submitting issues or feature requests to this project, you acknowledge that these suggestions may be implemented by the project maintainers without attribution or compensation.

## 🧪 Testing

### 🏗️ Current Test Structure

Tests are split across the feature/integration test modules and two files for pure helpers:

- **`tests/test_<area>.py`** (one `Test<Area>` class per module) — feature and integration tests, including all LLM-backed extraction tests that replay through VCR cassettes. Most contributions belong here, in the module of the relevant feature area:

  | Module | Feature area |
  |--------|--------------|
  | `test_aspects_and_concepts.py` | Aspects, concepts, examples and extracted items; extraction of concepts of specific types |
  | `test_chat.py` | Messages, chat sessions, chat with LLMs and tool calling |
  | `test_decision_models.py` | Decision models: questions for concepts and aspects, conversion of answers to item values and probabilities; extraction of aspects and concepts with decision models, on their own, in model groups and in fallback chains |
  | `test_documents.py` | Documents, paragraphs, sentences and extraction pipelines; text segmentation with SaT models |
  | `test_extraction.py` | Extraction of aspects and concepts from documents; extraction errors |
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
- **`tests/test_properties.py`** — Hypothesis-based property tests for pure utility helpers (text cleaning, JSON parsing, type-hint serialization, list chunking, etc.). No LLM, no shared test objects, no cassettes.
- **`tests/test_units.py`** — example-based unit tests for pure utility helpers (tool-schema generation, type-system normalization, error branches). Pins specific input/output contracts and error messages that property tests aren't well suited to express. No LLM, no shared test objects, no cassettes.

### 📏 Testing Guidelines

- Write tests for new features or bug fixes
- Make sure all tests pass before submitting a PR
- Maintain code coverage above **80%**
- Check code coverage by running:

  ```bash
  uv run pytest --cov=contextgem
  ```

---

### 🎲 Tests for Pure Helpers (Property-Based and Example-Based)

Pure utility helpers in `contextgem/internal/` (text cleaning, JSON parsing, type-hint serialization, tool-schema generation, etc.) are covered by two complementary test files. Both run without LLMs, shared test objects, or cassettes.

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

### 📼 VCR Cassette Management

We use [pytest-recording](https://github.com/kiwicom/pytest-recording) to record and replay HTTP interactions with LLM APIs (both cloud-based and local). This allows tests that call LLM APIs to run without making actual API calls after the initial recording.

#### Why VCR Cassettes?

VCR cassettes provide the most reliable testing approach for ContextGem because:

- **Real API Testing**: Testing with actual LLM APIs ensures our functionality works as expected with real responses, edge cases, and API behaviors
- **Scalability**: With a significant number of LLM API tests, hardcoding requests/responses would be impractical and unmaintainable
- **Reproducibility**: Once recorded, tests run consistently without variability in LLM responses
- **No Setup Friction**: Contributors can run tests without API keys or local LLM installations

Local LLMs (Ollama, LM Studio, etc.) also use HTTP APIs (typically on localhost) and their interactions are recorded in cassettes too.

The test suite automatically uses dummy environment variables with pre-recorded cassettes when no `.env` file is present, so most contributors won't need to set up real API keys or local LLM servers. The first test run downloads a small SaT segmentation model from HuggingFace, so it needs network access.

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

1. **Create a `.env` file** locally (ignored by git) with the API keys for the LLM services your new tests will use:

   ```bash
   # Only include the variables for LLM APIs your tests actually call
   
   # For OpenAI API tests
   CONTEXTGEM_OPENAI_API_KEY=your_openai_api_key
   
   # For TypeSafe API tests (decision models)
   CONTEXTGEM_TYPESAFE_API_KEY=your_typesafe_api_key
   
   # For Microsoft Foundry API tests (decision models)
   CONTEXTGEM_AZURE_AI_API_KEY=your_foundry_api_key
   CONTEXTGEM_AZURE_AI_API_BASE=https://your-resource.services.ai.azure.com
   ```

2. **For new LLM providers**, create environment variables prefixed with `CONTEXTGEM_`:

   ```bash
   CONTEXTGEM_GOOGLE_AI_STUDIO_API_KEY=your_google_api_key
   ```

   Also add the API host of the provider to `_TETHERED_RECORDING_ALLOW` in `tests/conftest.py`, otherwise its requests are blocked while recording (see [Network Egress Control](#network-egress-control)). If the provider does not send its key in the `Authorization` or `api-key` header (e.g. Google AI Studio sends it in `x-goog-api-key`), add that header to `VCR_FILTER_HEADERS` in `tests/utils.py`, otherwise the key is recorded in the cassette.

3. **Update dummy variables** in `tests/utils.py` by adding your new environment variables to the `default_env_vars` dictionary in `set_dummy_env_variables_for_testing_from_cassettes()`, mapped to a dummy value (e.g. "DUMMY")

4. **Add the VCR decorator** to your new test methods that call LLM APIs (cloud or local):

   ```python
   @pytest.mark.vcr
   def test_your_new_llm_feature(self):
       # Your test code that calls LLM APIs (cloud or local)
   ```

   > ⚠️ **Important:** Without the `@pytest.mark.vcr` decorator, no cassette will be recorded!

5. **Run your new tests** - new cassettes will be created automatically

6. **Verify redaction** - check that sensitive data is properly redacted in the new cassette files

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

> ⚠️ **Important:** This will use significant API quota and may incur substantial costs!

---

#### Environment Variable Security

**What is redacted automatically:**

- `CONTEXTGEM_OPENAI_API_KEY`, `CONTEXTGEM_TYPESAFE_API_KEY` and `CONTEXTGEM_AZURE_AI_API_KEY`: the headers that carry these keys (`Authorization`, `api-key`) are among the headers listed in `VCR_FILTER_HEADERS` (`tests/utils.py`), whose values are redacted in the recorded requests and responses. A key that is sent in another header, in the URL or in the body of a request is not redacted.
- `CONTEXTGEM_AZURE_AI_API_BASE`: the host of a Microsoft Foundry resource, which has the name of the resource, is replaced with a dummy host (`dummy-endpoint.local`) in the recorded requests, and the name of a project of the resource with a dummy name, if the endpoint has it

The response headers in which Azure names the deployment, the region and the cluster that served a request are redacted too. The name of your home directory is also redacted in the file paths found in the recorded responses, e.g. in the paths of the model files that a local LLM server returns.

#### Local LLM Testing

For local LLM testing, install the following tools and download the relevant models identified under `ollama` and `lm_studio` prefixes in `tests/test_llms.py`:

- [Ollama](https://ollama.com/)
- [LM Studio](https://lmstudio.ai/)

> ⚠️ **Important:** Your system needs to have an appropriate GPU capacity to run such local LLMs.

#### Decision Model Testing

Most tests of decision models in `tests/test_decision_models.py` replace the API call of `litellm` with a stub and need no cassettes. The tests that send requests to the APIs of decision models are marked with `@pytest.mark.vcr`: the ones in `tests/test_decision_models.py`, and `test_decision_model_usage_examples` in `tests/test_usage_examples.py`, which runs the usage examples in `dev/usage_examples/docs/decision_models/`. Recording their cassettes needs:

- **`CONTEXTGEM_TYPESAFE_API_KEY`** in your `.env` file, for the requests to TypeSafe (`typesafe/jev-latest`).
- **`CONTEXTGEM_OPENAI_API_KEY`** in your `.env` file, for the requests to the Decisions API of OpenAI (`openai/gpt-6-luna` as a decision model), and to the LLM that is used next to the decision model in a model group.
- **`CONTEXTGEM_AZURE_AI_API_KEY` and `CONTEXTGEM_AZURE_AI_API_BASE`** in your `.env` file, for the requests to a decision model deployed in Microsoft Foundry: the key and the endpoint of your Foundry resource (`https://<resource>.services.ai.azure.com`). The resource needs a deployment of Microsoft-Decision-1 named `ms-decision-model`: a model deployed in Microsoft Foundry is requested by the name of its deployment, which is the one in `tests/test_decision_models.py` (`azure_ai/ms-decision-model`). It must have no deployment named `Microsoft-Decision-1`, which one of the tests requests to get the error for a deployment that does not exist.
- **[Ollama](https://ollama.com/) 0.35 or later**, running on its default port, with the decision model identified under the `strands_decider/` prefix in `tests/test_decision_models.py` downloaded (`ollama pull nimble`), for the requests to a local decision model. No API key is needed.
- **[LM Studio](https://lmstudio.ai/) 0.4.26 or later** (not LM Studio Bionic, whose server does not serve decision models), running its local server on the default port, with the decision model of the LM Studio test in `tests/test_decision_models.py` downloaded (`lms get https://huggingface.co/ggml-org/Kev-4B-GGUF`). No API key is needed.

`litellm` rejects a request to a decision model that has no API key before the request is sent, i.e. before it can be replayed from a cassette. For this reason, a dummy `CONTEXTGEM_TYPESAFE_API_KEY` is set when no `.env` file is present. If your `.env` file has no `CONTEXTGEM_TYPESAFE_API_KEY`, add the variable with any value, otherwise these tests fail also in replay.

A decision model deployed in Microsoft Foundry is requested with a key and with the endpoint of the resource. A dummy `CONTEXTGEM_AZURE_AI_API_KEY` and a dummy `CONTEXTGEM_AZURE_AI_API_BASE` (`https://dummy-endpoint.local`, the host that the host of a resource is replaced with in the cassettes) are set when no `.env` file is present. The tests use the same dummy values if your `.env` file has no key or endpoint of a Foundry resource, so these tests replay without the two variables.

The host of TypeSafe (`api.typesafe.ai`) is on the list of the hosts allowed while recording. The host of your Microsoft Foundry resource is allowed while recording as well: it differs for each resource, so it is read from `CONTEXTGEM_AZURE_AI_API_BASE`, and is accepted only if the endpoint is an HTTPS one of a host under `services.ai.azure.com`. A test that sends requests to another provider of decision models needs the host of that provider added to that list (see [Network Egress Control](#network-egress-control)).

#### Important Notes

> **💰 Cost Warning:** Recording cassettes for test methods that use live LLM API (non-local LLMs) uses your API keys and **will incur charges**. Scenario 4 (re-recording all cassettes) can be particularly expensive.
>
> **🔒 Security:** API keys are redacted in cassettes only where they are sent in the headers listed in `VCR_FILTER_HEADERS` (`tests/utils.py`), so always verify new cassette content.
>
> **🧪 Testing:** After recording, delete your `.env` file and run tests again to ensure LLM API tests pass by replaying from cassettes with dummy variables.
>
> **📼 Cassette names:** Cassette files are named `<TestClass>.<test name>.yaml`. If you move a test that calls LLM APIs to another test class, rename its cassette files accordingly (with `git mv`) in the same change. Otherwise, the test will not find its cassette and will record a new one.

#### Network Egress Control

ContextGem uses [tethered](https://github.com/shcherbak-ai/tethered) for network egress control in **two complementary places**:

**1. Production code (`tethered.scope`)** — the two non-LLM paths the framework controls are wrapped with `tethered.scope`, which can only narrow (intersection-only) egress and never widens a host application's policy:

- `contextgem/internal/utils.py::_load_sat_model` — restricted to HuggingFace hosts (`_SAT_MODEL_DOWNLOAD_HOSTS`)
- `contextgem/internal/base/models.py::_calculate_auto_pricing_costs` — restricted to `raw.githubusercontent.com` (`_GENAI_PRICES_REFRESH_HOSTS`)

LLM API calls are deliberately **not** scoped — endpoints are user-configured (Azure custom domains, OpenAI-compatible proxies, self-hosted, etc.) and are left to the host application's own egress policy.

**2. Test suite (`tethered.activate`)** — the `_tethered_network_guard` fixture in `tests/conftest.py` enforces egress control at the socket level during every test:

- **Tests without `@pytest.mark.vcr`**: all outbound connections are blocked, including the ones to localhost
- **VCR-marked tests, replay mode** (cassette exists): blocks all outbound connections except HuggingFace (for SaT model downloads not captured by VCR)
- **VCR-marked tests, recording mode** (no cassette): allows only approved endpoints (LLM APIs, HuggingFace for model downloads, genai-prices for cost data), the Microsoft Foundry resource whose endpoint is set in `CONTEXTGEM_AZURE_AI_API_BASE`, and localhost for local LLMs

The fixture activates tethered in **hardened mode** (`locked=True` with a private lock token, `external_subprocess_policy="block"`), so test code can't disable the policy mid-test and non-Python subprocesses are refused — closing the shell-egress bypass.

The policy above applies while a test runs. Before the tests start, the `_warm_token_encodings` session fixture counts tokens once, so that the token encodings used by `litellm` are available during the tests. On Windows, this downloads the encodings if `tiktoken` rejects the bundled ones (see the Windows note in [Running Tests](#-running-tests)). The download is wrapped with `tethered.scope` that allows a single host (`openaipublic.blob.core.windows.net`), which is deliberately not a part of the replay and recording allowlists, and `tiktoken` verifies each downloaded encoding against the hash pinned in its package.

The host allowlists in `tests/conftest.py` are sourced from the production constants (`_SAT_MODEL_DOWNLOAD_HOSTS`, `_GENAI_PRICES_REFRESH_HOSTS`) so the test allowlist cannot drift from the per-call production scopes. If you add tests that connect to new endpoints (other than these two), update the `_TETHERED_RECORDING_ALLOW` list in `tests/conftest.py`.

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
- Concurrency optimization warnings (testing performance comparisons)
- Deprecation warnings from dependencies

**Key Point**: If tests **PASS** with warnings, this should **not** prevent you from submitting your PR. The test suite is designed to handle and expect these warnings as part of normal operation.

### 🐞 Debugging Tests

The tests run with the ContextGem logger at the `DEBUG` level, which is set in `tests/conftest.py`, so the log output shows detailed information about test execution.

## 📚 Documentation

- Update documentation for any changed functionality
- Document new features
- Use clear, concise language

### 🏗️ Building the Documentation

Use the fabric commands from the project root:

#### For Live Development (Recommended)

```bash
# Live rebuild with auto-refresh on file changes
uv run fab docs-live
```

This starts a development server on `http://localhost:9000` with:

- Automatic rebuilds when files change
- Browser auto-refresh
- Pretty URLs without `.html` extensions

#### For Static Builds

```bash
# Build with verbose output, ignore cache, and treat warnings as errors
uv run fab docs
```

### 👀 Viewing the Documentation

**With Live Development:**
Open `http://localhost:9000` in your browser.

**With Static Builds:**
The pages of a static build link to each other as directories, which a browser does not resolve when the files are opened from disk. After building, serve the folder and open `http://localhost:9000`:

```bash
uv run python -m http.server 9000 --directory docs/build/dirhtml
```

### 🌐 Live Documentation

You can access the live documentation at: <https://contextgem.dev>

> **Note:** Documentation is automatically deployed when maintainers merge changes from `dev` to `main`. As a contributor, your documentation changes will be visible on the live site after your PR is merged and subsequently deployed by maintainers.

### 📋 Documentation Structure

Under `docs/`:

- `source/` - Contains the source `.rst` files
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
