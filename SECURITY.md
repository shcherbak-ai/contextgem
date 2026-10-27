# Security Policy

## Supported Versions

Security fixes are released in a new version of ContextGem 1.x, and earlier releases are not patched: upgrade to the latest release to get them. ContextGem 0.x gets no more releases. To upgrade from it, follow the [migration guide](https://contextgem.shcherbak.ai/migration_v1/).

## Reporting a Vulnerability

Report a vulnerability privately, by email to `sergii@shcherbak.ai`, not in a public issue. Please include:

- a description of the issue and of what it allows
- the steps to reproduce it
- the versions of ContextGem and Python, and any relevant logs or configuration, without API keys

Reports are answered as promptly as possible. There is no bug bounty program. If you are unsure whether something is a vulnerability or a bug, ask by email first.

Report a vulnerability in a dependency, e.g. LiteLLM, to that project. If the version ranges of ContextGem allow an affected version, tell ContextGem too.

## How the Project Is Checked

- **Code:** [CodeQL](https://codeql.github.com/) analyses the code on each push and pull request to `main` and `dev`, and once a week. [Bandit](https://github.com/PyCQA/bandit) checks it in a pre-commit hook, which CI also runs on each push and pull request.
- **Dependencies:** a new release of a dependency enters the lockfile only once it is 7 days old, by when a compromised release has usually been found and removed. At each commit, each locked version, Python and npm, and the version of uv that locks them, is checked against the advisories of [osv.dev](https://osv.dev). A report of a malicious package stops the commit, as does a known vulnerability, unless it is accepted with its reason, e.g. when no version that the ranges allow fixes it yet.
- **Workflows:** each GitHub Action is pinned to the SHA of a commit, and each workflow has read-only permissions, but for the jobs that need more. No checkout leaves its token to the later steps, and the builds that are published, of the release and of the documentation site, restore no cache, which a job of another workflow could have written.
- **Releases:** each release is built and published on GitHub Actions, through PyPI's trusted publishing, with an attestation of the workflow that built each file, which PyPI shows on the page of the file. Before the upload, the wheel is installed in a fresh environment with the locked versions of its dependencies, checked by their hashes.

## Dependencies

ContextGem builds on open-source packages, such as LiteLLM and Pydantic. Packages on PyPI are a target of supply-chain attacks, in which an attacker who gains a maintainer's access uploads a malicious release: LiteLLM had three such releases in 2026, all removed from PyPI since. [Dependency Security](https://contextgem.shcherbak.ai/dependency_security/) in the documentation shows how to keep the dependencies of your application under control, with a lockfile, new releases only once they are a week old and checks against known advisories, and what ContextGem does on its side.

## Data Privacy

ContextGem collects no telemetry. It sends your documents, and the prompts built from them, only to the models that you configure, LLMs and decision models alike, through [LiteLLM](https://github.com/BerriAI/litellm), which runs as a library in your environment. According to [LiteLLM's documentation](https://docs.litellm.ai/docs/data_security), no data is stored on LiteLLM's servers when it runs in your environment, and its usage telemetry is off by default.

Other requests download public data and send none of your documents, for example:

- LiteLLM's list of model prices, from GitHub, when LiteLLM is imported (set `LITELLM_LOCAL_MODEL_COST_MAP=True` to use the copy in the package)
- the SaT models that split text into sentences, from Hugging Face, on first use
- the prices of genai-prices, from GitHub, with `auto_pricing_refresh=True`
