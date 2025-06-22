# Security Policy


## Supported Versions

We maintain security practices for the latest release of this library. Older versions may not receive security updates.


## Security Testing

This project is automatically tested for security issues using multiple security scanning tools:

- **[CodeQL](https://codeql.github.com/)** - GitHub's semantic code analysis engine for vulnerability detection (run via GitHub Actions)
- **[Bandit](https://github.com/PyCQA/bandit)** - Python security linter for common security issues (run via GitHub Actions and pre-commit hooks)
- **[Snyk](https://snyk.io)** - Dependency vulnerability monitoring (used as needed)


## Data Privacy

This library uses LiteLLM as a local Python package to communicate with LLM providers using unified interface. No data or telemetry is transmitted to LiteLLM servers, as the SDK is run entirely within the user's environment. According to LiteLLM's documentation, self-hosted or local SDK use involves no data storage and no telemetry. For details, see [LiteLLM's documentation](https://docs.litellm.ai/docs/data_security).


## Reporting a Vulnerability

We value the security community's role in protecting our users. If you discover a potential security issue in this project, please report it as follows:

📧 **Email**: `sergii@shcherbak.ai`

When reporting, please include:
- A detailed description of the issue
- Steps to reproduce the vulnerability
- Any relevant logs, context, or configurations

We aim to respond promptly to all valid reports. Please note that we do not currently offer a bug bounty program.


## Questions?

If you're unsure whether something is a vulnerability or just a bug, feel free to reach out via the email above before submitting a full report.
