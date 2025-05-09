# Changelog
All notable changes to ContextGem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with the following additional categories:

- **Refactor**: Code reorganization that doesn't change functionality but improves structure or maintainability

## [0.2.4](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.2.4) - 2025-05-09
### Fixed
- Removed 'think' tags and content from LLM outputs (e.g. when using DeepSeek R1 via Ollama) which was breaking JSON parsing and validation

### Added
- Documentation for cloud/local LLMs and LLM configuration guide

## [0.2.3](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.2.3) - 2025-05-04
### Changed
- Updated litellm dependency version after encoding bug has been fixed upstream

## [0.2.2](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.2.2) - 2025-05-02
### Refactor
- Refactored DOCX converter internals for better maintainability

## [0.2.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.2.1) - 2023-04-30
### Fixed
- Fixed litellm dependency issue, pinning to version ==1.67.1 to avoid encoding bug in newer versions of litellm

## [0.2.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.2.0) - 2023-04-21
### Added
- Added DocxConverter for converting DOCX files into ContextGem Document objects