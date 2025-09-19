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

    # Install dependencies and development extras
    uv sync --all-groups
    ```

3. **🔧 Install pre-commit hooks**:
    ```bash
    # Install pre-commit hooks
    uv run pre-commit install

    # Install commit-msg hooks (for commitizen)
    uv run pre-commit install --hook-type commit-msg
    ```


### 📁 Project Structure

Below is a high-level overview of the codebase layout and where to make different types of contributions:

```
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
│       └── ...                   #   - More public modules
│
├── tests/
│   ├── cassettes/                # 📼 VCR recordings (auto-generated)
│   ├── test_all.py               # ✅ Add your tests here
│   ├── utils.py                  # 🛠️ Test utilities & dummy env vars
│   └── ...                       # 📁 Test data files
│
├── docs/
│   ├── source/                   # 📚 Documentation source files
│   └── ...                       # 📋 Build configs & outputs
│
├── dev/
│   ├── usage_examples/           # 📝 Code examples for docs
│   ├── notebooks/                # 📓 Notebooks (auto-generated)
│   ├── readme.template.md        # ✏️ Edit this, not README.md
│   └── ...                       # 🛠️ Development scripts
│
├── pyproject.toml                # ⚙️ Dependencies & project config
└── README.md                     # 🤖 Auto-generated (don't edit)
```

**🎯 Quick Start for Your Contribution:**
- **Adding new functionality?** → Implement in `contextgem/internal/` (core logic). Then expose via a thin public facade in `contextgem/public/` using the registry.
- **Writing tests?** → Add to `tests/test_all.py::TestAll`  
- **Updating docs?** → Edit files in `docs/source/` or `dev/`
- **Fixing README?** → Edit `dev/readme.template.md`

> **💡 Note:** Implement functionality in `internal/` (base classes, validation, serialization, typing). Use `public/` to expose thin, documented facades that inherit from internal classes and are registered with `@_expose_in_registry` decorator to ensure deserialization and instance creation utils return public types. Do not import public classes in internal modules; use the registry for type resolution and publicization.


---

### ✏️ Making Changes

1. **🌿 Create a new branch**:

    For example:
    ```bash
    git checkout -b feature/your-feature-name
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
    - **Pyright**: For static type checking
    - **Bandit**: For Python security vulnerability scanning
    - **Deptry**: For dependency health checks (unused, missing, transitive dependencies)
    - **Interrogate**: For docstring coverage checking
    - **Pre-commit hooks**: To automatically check and format code before commits

    The pre-commit hooks will automatically check and format your code when you commit. There are two scenarios to be aware of:

    **If the hooks modify any files during commit** (such as Ruff formatting):
    1. Review the changes made
    2. Add the modified files to the staging area
    3. Commit again

    **If security issues are detected** (Bandit):
    1. Review the security findings in the terminal output
    2. Fix the identified security issues in your code
    3. Add the fixed files to the staging area
    4. Commit again

3. **🧪 Run tests** to ensure your changes do not break existing functionality:
   ```bash
   uv run pytest
   ```

   > **Note:** We use [pytest-recording](https://github.com/kiwicom/pytest-recording) to record and replay LLM API interactions. Your changes may require re-recording VCR cassettes for the tests. See [VCR Cassette Management](#vcr-cassette-management) section below for details.

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
   ```
   docs(readme): update installation instructions
   ```

   > **Note:** If pre-commit hooks fail or modify files during `cz commit`, you can retry with the same message:
   > ```bash
   > uv run cz commit --retry
   > ```


## 🔄 Pull Request Process

1. **🔄 Update your fork** with the latest changes from the `dev` branch:
   ```bash
   git fetch upstream
   git checkout dev
   git merge upstream/dev
   git push origin dev
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

Currently, all tests are located in a single file: `tests/test_all.py` within the `TestAll` class. When adding new tests, place them in this file following the existing patterns.

> **Note:** We plan to refactor tests into multiple files for better maintainability in the future, but for now all tests should be added to `tests/test_all.py`.

### 📏 Testing Guidelines

- Write tests for new features or bug fixes
- Make sure all tests pass before submitting a PR
- Maintain code coverage above **80%**
- Check code coverage by running:
  ```bash
  uv run pytest --cov=contextgem
  ```

---

### 📼 VCR Cassette Management

We use [pytest-recording](https://github.com/kiwicom/pytest-recording) to record and replay HTTP interactions with LLM APIs (both cloud-based and local). This allows tests that call LLM APIs to run without making actual API calls after the initial recording.

> **Note:** Tests that do not call LLM APIs do not require or use VCR cassettes. The cassette system only applies to tests that interact with LLM APIs.

#### Why VCR Cassettes?

VCR cassettes provide the most reliable testing approach for ContextGem because:

- **Real API Testing**: Testing with actual LLM APIs ensures our functionality works as expected with real responses, edge cases, and API behaviors
- **Scalability**: With a significant number of LLM API tests, hardcoding requests/responses would be impractical and unmaintainable
- **Reproducibility**: Once recorded, tests run consistently without variability in LLM responses
- **No Setup Friction**: Contributors can run tests without API keys or local LLM installations

Local LLMs (Ollama, LM Studio, etc.) also use HTTP APIs (typically on localhost) and their interactions are recorded in cassettes too.

The test suite automatically uses dummy environment variables with pre-recorded cassettes when no `.env` file is present, so most contributors won't need to set up real API keys or local LLM servers.

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
   ```
   # Only include the variables for LLM APIs your tests actually call
   
   # For OpenAI API tests
   CONTEXTGEM_OPENAI_API_KEY=your_openai_api_key
   
   # For Azure OpenAI tests
   CONTEXTGEM_AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   CONTEXTGEM_AZURE_OPENAI_API_BASE=your_azure_openai_base
   CONTEXTGEM_AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
   
   # For debugging output
   CONTEXTGEM_LOGGER_LEVEL=DEBUG
   ```

2. **For new LLM providers**, create environment variables prefixed with `CONTEXTGEM_`:
   ```
   CONTEXTGEM_GOOGLE_AI_STUDIO_API_KEY=your_google_api_key
   ```

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
   uv run pytest tests/test_all.py::TestAll::test_specific_method
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

2. **Create a `.env` file** with your API keys (same as Scenario 2)

3. **Run all tests** to re-record everything:
   ```bash
   uv run pytest
   ```

> ⚠️ **Important:** This will use significant API quota and may incur substantial costs!

---

#### Environment Variable Security

**Automatically Redacted Variables:**
- `CONTEXTGEM_OPENAI_API_KEY`
- `CONTEXTGEM_AZURE_OPENAI_API_KEY`
- `CONTEXTGEM_AZURE_OPENAI_API_BASE`
- `CONTEXTGEM_AZURE_OPENAI_API_VERSION`

**Adding New Variables:**
- Use the `CONTEXTGEM_` prefix for new API keys
- Verify redaction in your cassette files
- Update redaction logic in `tests/utils.py` if needed
- Add dummy values to `set_dummy_env_variables_for_testing_from_cassettes()`

#### Local LLM Testing

For local LLM testing, install the following tools and download the relevant models identified under `ollama` and `lm_studio` prefixes in `tests/test_all.py`:
- [Ollama](https://ollama.ai/) 
- [LM Studio](https://lmstudio.ai/)
> ⚠️ **Important:** Your system needs to have an appropriate GPU capacity to run such local LLMs.

#### Important Notes

> **💰 Cost Warning:** Recording cassettes for test methods that use live LLM API (non-local LLMs) uses your API keys and **will incur charges**. Scenario 4 (re-recording all cassettes) can be particularly expensive.

> **🔒 Security:** Environment variables such as API keys are automatically stripped from cassettes, but always verify new cassette content.

> **🧪 Testing:** After recording, delete your `.env` file and run tests again to ensure LLM API tests pass by replaying from cassettes with dummy variables.

#### URL Security Validation

The test suite includes automated security validation for VCR cassettes to ensure that only approved domains are accessed during testing. This helps maintain security by ensuring tests connect only to explicitly authorized endpoints.

The URL security check validates that all URLs in cassette files are from approved domains (such as `api.openai.com`, `localhost` for local LLMs, etc.). If you add tests that connect to new endpoints, you may need to update the approved domains list in `tests/url_security.py`.

> **Note:** URL security validation is automatically skipped on Windows *when running with coverage* to avoid access violations that occur when processing large YAML cassette files under coverage instrumentation.


---

### 🏃 Running Tests

Run all tests:
```bash
uv run pytest
```

Run specific tests:
```bash
# Run a specific test method
uv run pytest tests/test_all.py::TestAll::test_extract_all
```

**🔍 Optional Memory Profiling**: For performance testing, you can enable memory profiling to analyze memory usage during test execution:
```bash
uv run pytest --mem-profile
```

> **Note:** Memory profiling adds significant overhead and tests will run much slower when profiling is enabled. Memory profiling helps ensure that ContextGem objects don't consume excessive memory and validates memory usage against defined reasonable limits.

### ⚠️✅ Expected Test Warnings

Warnings generated during tests are often expected and by design. Many warnings are intentionally triggered to test error handling, edge cases, and warning systems. Common expected warnings include:

- LLM extraction errors and retries (testing error handling)
- Missing LLM roles (testing validation logic)
- Concurrency optimization warnings (testing performance comparisons)
- Deprecation warnings from dependencies

**Key Point**: If tests **PASS** with warnings, this should **not** prevent you from submitting your PR. The test suite is designed to handle and expect these warnings as part of normal operation.

### 🐞 Debugging Tests

The log output will show detailed information about test execution.


## 📚 Documentation

- Update documentation for any changed functionality
- Document new features
- Use clear, concise language

### 🏗️ Building the Documentation

Navigate to the `docs/` directory and choose your preferred build method:

#### For Live Development (Recommended)

Use `sphinx-autobuild` for live reloading during development:

```bash
# Live rebuild with auto-refresh on file changes
make livehtml
# Or on Windows: ./make.bat livehtml
```

This starts a development server on `http://localhost:9000` with:
- Automatic rebuilds when files change
- Browser auto-refresh
- Pretty URLs without `.html` extensions

#### For Static Builds

For one-time builds or CI-style building:

```bash
# Build with verbose output, ignore cache, and treat warnings as errors 
# (recommended for structural changes)
uv run sphinx-build -b dirhtml source build/dirhtml -v -E -W
```

The `-E` flag ensures Sphinx completely rebuilds the environment, which is especially important after structural changes like modifying toctree directives or removing files. The `dirhtml` format creates pretty URLs without `.html` extensions, consistent with the live development server.

### 👀 Viewing the Documentation

**With Live Development:**
The documentation automatically opens at `http://localhost:9000` when using `make livehtml`.

**With Static Builds:**
After building, open `build/dirhtml/index.html` in your web browser to view the documentation.

### 🌐 Live Documentation

You can access the live documentation at: https://contextgem.dev

> **Note:** Documentation is automatically deployed when maintainers merge changes from `dev` to `main`. As a contributor, your documentation changes will be visible on the live site after your PR is merged and subsequently deployed by maintainers.

### 📋 Documentation Structure

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
# Populate README.md from template
python dev/populate_project_readme.py
```


## ❓ Questions & Support

We're here to help! Whether you're stuck on something technical, have questions about the contribution process, or want to suggest improvements to this guide, don't hesitate to reach out.

### 🆘 Get Help With:
- **Technical Issues**: Setup problems, test failures, or development environment issues
- **Contribution Process**: Questions about pull requests, branching, or code review
- **Feature Ideas**: Discussion about new features or improvements
- **Documentation**: Clarifications about this contributing guide or suggesting improvements

### 📞 Contact Options:

**🐛 GitHub Issues** (preferred for technical questions):
- [Open a new issue](https://github.com/shcherbak-ai/contextgem/issues/new) using our issue templates

**📧 Direct Contact**:
- **📧 Email**: sergii@shcherbak.ai
- **💼 LinkedIn**: [Sergii Shcherbak](https://www.linkedin.com/in/sergii-shcherbak-10068866/)
- **🐦 X**: [@seshch](https://x.com/seshch)

### 📖 Improving This Guide

Found something unclear in this contributing guide? Missing information that would have helped you? Please:
- Open an issue with the `documentation` label
- Suggest specific improvements or additions
- Share your contributor experience to help us improve the process

---

**Thank you for contributing to ContextGem!** 🙏

Your contributions help make ContextGem better for everyone. We appreciate your time and effort!
