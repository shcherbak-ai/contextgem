# Changelog
All notable changes to ContextGem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with the following additional categories:

- **Refactor**: Code reorganization that doesn't change functionality but improves structure or maintainability

## [0.11.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.11.0) - 2025-07-10
### Added
- Support for litellm versions >1.71.1: ContextGem now supports newer litellm versions that were previously incompatible with tests due to underlying transport changes (removal of httpx-aiohttp dependency) introduced after v1.71.1, which affected VCR recording used in testing.

## [0.10.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.10.0) - 2025-07-07
### Added
- RatingConcept now supports tuple format for rating scales: Use `(start, end)` tuples instead of `RatingScale` objects for simpler API. Example: `rating_scale=(1, 5)` instead of `rating_scale=RatingScale(start=1, end=5)`.

### Deprecated
- RatingScale class is deprecated and will be removed in v1.0.0. Use tuple format `(start, end)` instead for rating scales in RatingConcept.

## [0.9.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.9.0) - 2025-07-02
### Added
- New exception handling for LLM extraction methods: Added `raise_exception_on_extraction_error` parameter (default is True) to LLM extraction methods. Controls whether to raise an exception when LLM returns invalid data (`LLMExtractionError`) or when there is an error calling LLM API (`LLMAPIError`). When False, warnings are issued and no extracted items are returned.

## [0.8.2](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.8.2) - 2025-06-25
### Changed
- Improved prompts

## [0.8.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.8.1) - 2025-06-23
### Added
- Documentation on troubleshooting issues with small models.

## [0.8.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.8.0) - 2025-06-22
### Changed
- Deferred SaT segmentation: SaT segmentation is now performed only when actually needed, improving both document initialization and extraction performance, as some extraction workflows may not require SaT segmentation.

## [0.7.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.7.0) - 2025-06-16
### Added
- DocxConverter upgrade: migrated to high-performance lxml library for parsing DOCX document structure, added processing of links and inline formatting, improved conversion accuracy.
- Integrated Bandit security scanning across development pipeline.

### Changed
- Updated documentation to reflect the above changes.

## [0.6.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.6.1) - 2025-06-04
### Changed
- Updated documentation for LM Studio models to clarify dummy API key requirement

## [0.6.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.6.0) - 2025-06-03
### Added
- LabelConcept - a classification concept type that categorizes content using predefined labels.

## [0.5.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.5.0) - 2025-05-29
### Fixed
- Params handling for reasoning (CoT-capable) models other than OpenAI o-series. Enabled automatic retry of LLM calls with dropping unsupported params if such unsupported params were set for the model. Improved handling and validation of LLM call params.

### Changed
- Migrated to wtpsplit-lite - a lightweight version of wtpsplit that only retains accelerated ONNX inference of SaT models with minimal dependencies.

## [0.4.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.4.1) - 2025-05-26
### Added
- Comprehensive docs on extracting aspects, extracting concepts, and LLM extraction methods

## [0.4.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.4.0) - 2025-05-20
### Added
- Support for local SaT model paths in Document's `sat_model_id` parameter

## [0.3.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.3.0) - 2025-05-19
### Added
- Expanded JsonObjectConcept to support nested class hierarchies, nested dictionary structures, lists containing objects, and literal types.

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