# Changelog

All notable changes to ContextGem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with the following additional categories:

- **Refactor**: Code reorganization that doesn't change functionality but improves structure or maintainability

## [1.0.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v1.0.0) - 2026-10-06

### Added

- Added the `modality` parameter to concepts, which sets the document content a concept is extracted from: `"text"` (default), `"image"`, or `"multimodal"` (text and/or images, whichever the document has). `"image"` and `"multimodal"` are supported for document-level concepts only. Previously, the content was determined by the role of the LLM (see "Changed").
- Added the `fallback_models` parameter to `DocumentLLM`, which accepts a list of fallback LLMs and replaces the `fallback_llm` parameter, which accepted a single one (see "Removed"). Fallback LLMs are tried in the given order until one of them succeeds, for both extraction and chat, e.g. to fall back across several providers during an outage. Each fallback LLM must have `is_fallback=True` and the same role, output language, and tool configuration as the primary LLM, must differ in configuration from the primary LLM and from the other fallback LLMs, and cannot have fallback LLMs of its own. Usage and cost are tracked for the primary LLM and each of its fallback LLMs.
- Added tracking of cached input tokens. The usage returned by `get_usage()` now includes `cache_read` and `cache_write`: the number of input tokens that the LLM provider reports as read from, and written to, its prompt cache. Both are a part of the `input` token count, and are zero if the provider does not report such tokens.
- Added the optional `cache_read_per_1m_tokens` and `cache_write_per_1m_tokens` parameters to `ModelPricing` (previously `LLMPricing`, see "Changed"), which set the prices of the input tokens read from and written to the prompt cache of the LLM provider. If a cache price is not set, such tokens are priced as regular input tokens, as before. Serialized `ModelPricing` data now includes these keys.

### Changed

- **Breaking:** Renamed the classes, parameters, and attributes that are not specific to LLMs, so that their names refer to models in general. `DocumentLLM` keeps its name. The old names are removed, and using them raises an error. See the [migration guide](https://contextgem.dev/migration_v1/) for updating the code and the serialized data.
  - `DocumentLLMGroup` is renamed to `DocumentModelGroup`, and its `llms` parameter and attribute to `models`.
  - The `llm_role` parameter and attribute of aspects and concepts is renamed to `model_role`, as is the `llm_role` argument of `get_usage()`, `get_cost()`, and `reset_usage_and_cost()` of `DocumentModelGroup`.
  - The `llm_roles` property of documents, pipelines, and aspects is renamed to `model_roles`.
  - `LLMPricing` is renamed to `ModelPricing`.
  - `LLMAPIError` is renamed to `ModelAPIError`, and `LLMExtractionError` to `ModelExtractionError`.
- **Breaking:** Serialized data uses the new names: the `model_role` key of aspects and concepts, the `models` key and the `DocumentModelGroup` class name of model groups, the `ModelPricing` class name of pricing details, and the class names of the usage and cost containers stored in serialized LLMs (always empty, as the usage and cost statistics are not serialized). Documents, pipelines, aspects, concepts, LLMs, and model groups serialized by earlier versions can no longer be deserialized as is. Update the serialized data as described in the [migration guide](https://contextgem.dev/migration_v1/).
- **Breaking:** Model roles are now free-form labels. The `role` parameter of `DocumentLLM` and the `model_role` parameter of aspects and concepts accept any label of up to 64 printable characters (the surrounding whitespace is removed; line breaks, tabs, and other control or invisible characters are not allowed) and default to `None` (no role), instead of one of the six predefined role names (`"extractor_text"`, `"reasoner_text"`, `"extractor_vision"`, `"reasoner_vision"`, `"extractor_multimodal"`, `"reasoner_multimodal"`), with `"extractor_text"` as the default. A role only routes aspects and concepts to the LLMs of a `DocumentModelGroup`: an aspect or concept is extracted by the LLM with the `role` equal to its `model_role`, and an aspect or concept without `model_role` is extracted by the LLM without `role`. A role no longer implies the capabilities of the model or the content it can process, so the same model no longer needs to be added to a group several times, with different roles, to process both text and images. Each LLM in a group must still have a unique role, i.e. only one LLM in a group can have no role. The predefined role names have no special meaning anymore: if still used, they are treated as any other role labels, and no longer determine the content a concept is extracted from. See the [migration guide](https://contextgem.dev/migration_v1/) for updating the code and the serialized data.
- **Breaking:** A `DocumentLLM` used on its own (not in a group) now extracts all the aspects and concepts it is given, regardless of their `model_role`. Previously, it skipped the aspects and concepts with a `model_role` different from its `role`, with a warning. To extract only some of the aspects or concepts with a single LLM, use the `from_aspects` / `from_concepts` parameters of the extraction methods.
- **Breaking:** A `DocumentModelGroup` now raises `ValueError` if an aspect or concept to be extracted has a `model_role` that no LLM in the group has. This is checked before sending any requests for that extraction method, including for the nested aspects and concepts that the method processes. Previously, such aspects and concepts were skipped: with a warning in `extract_all()`, and silently in the other extraction methods.
- **Breaking:** The content a concept is extracted from is now determined by the `modality` of the concept instead of the role of the LLM. Concepts with different modalities can be extracted by the same LLM, in separate calls.
- Vision and reasoning capabilities of a model are no longer checked against the role of the LLM when the LLM is created, i.e. the related warnings are no longer issued. Instead, `ValueError` is raised before any requests are sent, if the LLM that is to extract concepts from images, or any of its fallback LLMs, is not detected as vision-capable (unless `_supports_vision=True` is set on the LLM). The same applies to the fallback LLMs when images are passed to `chat()`. Previously, such error was raised only when the images were sent to the model, i.e. for a fallback LLM only after the primary LLM failed. As before, the reasoning-specific prompt instructions are used based on the detected reasoning support of the model. The warning about the `ollama_chat/` prefix not supporting image inputs is now issued when images are sent to such a model, instead of when the LLM is created.
- The `model_role` argument of `get_usage()`, `get_cost()`, and `reset_usage_and_cost()` of `DocumentModelGroup` is now validated in the same way as the roles assigned to LLMs: the surrounding whitespace is removed, and an invalid role, e.g. an empty string, raises `ValueError`. Previously, the argument was compared as passed, and an empty string selected all LLMs.
- The `model_roles` property of documents, pipelines, and aspects now also includes the roles assigned to sub-aspects and their concepts, and contains `None` if there are aspects or concepts without a role.
- Serialized concept data now includes the `modality` key. Documents and pipelines serialized by earlier versions store the predefined role names and no modality, and need to be updated as described in the [migration guide](https://contextgem.dev/migration_v1/).
- **Breaking:** Serialized `DocumentLLM` data now stores fallback LLMs under the `fallback_models` key instead of the `fallback_llm` key. An LLM or model group serialized by earlier versions has the `fallback_llm` key in each LLM, even if no fallback LLM was assigned, and can no longer be deserialized as is. Update the serialized data as described in the [migration guide](https://contextgem.dev/migration_v1/), or create the LLMs anew.
- Failed LLM API requests are now retried by ContextGem instead of litellm, with `num_retries_failed_request` keeping its meaning. Only transient errors (request timeout, rate limit, server or connection error) are retried; non-transient errors (e.g. invalid request, failed authentication, exceeded context window) fail right away or are passed to a fallback LLM, instead of being retried as before. Retries are performed with exponential backoff and jitter (about 1, 2, 4, 8 seconds, capped at 10 seconds, each randomly shortened by up to a half); previously, only rate limit errors were retried with a backoff, without jitter, and other errors were retried immediately.
- `ModelAPIError.retry_count`, as well as the corresponding warning messages, now report the number of retries actually attempted for the failed request, rather than the configured number of retries.
- The warning issued when the pricing data of a model is not found with `auto_pricing=True` now also suggests setting `auto_pricing_refresh=True`, unless it is already set, as the pricing data bundled with `genai-prices` may not include a recently released model.
- Upgraded pinned dependency versions: `litellm==1.104.0`, `genai-prices==0.1.9`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases. `openai` remains pinned to `2.54.0`, the latest version supported by `litellm`. As of this version, `litellm` installs `boto3` as a required dependency.

### Removed

See the [migration guide](https://contextgem.dev/migration_v1/) for updating the code and the serialized data affected by the removals.

- **Breaking:** Removed the `fallback_llm` parameter and attribute of `DocumentLLM`. Use `fallback_models` instead, e.g. `fallback_models=[fallback_llm]`. Passing `fallback_llm` to `DocumentLLM`, or reading or assigning the attribute, now raises an error.
- **Breaking:** Removed `DocumentPipeline`, deprecated since v0.14.1. Use `ExtractionPipeline` instead, which has the same parameters and methods. A pipeline serialized as `DocumentPipeline` can no longer be deserialized as is.
- **Breaking:** Removed `RatingScale`, deprecated since v0.10.0. Pass the rating scale of a `RatingConcept` as a tuple of `(start, end)` integers instead, e.g. `rating_scale=(1, 5)`. Documents, pipelines, aspects, and concepts serialized with a rating scale defined as a `RatingScale` object can no longer be deserialized as is.
- **Breaking:** Removed `DocxConverter`, deprecated since v0.22.0, together with the DOCX converter exceptions (`DocxConverterError` and its subclasses). Use a dedicated document conversion library (e.g. [Docling](https://github.com/docling-project/docling), [MarkItDown](https://github.com/microsoft/markitdown)) to convert files to text, then pass the result to `Document(raw_text=...)`.
- **Breaking:** Removed the Markdown text that `DocxConverter` attached to documents and paragraphs and that was sent to the LLM instead of the plain text. Text is now always extracted from `raw_text`. Documents converted from DOCX and serialized by earlier versions can still be deserialized: their Markdown text is ignored and is not kept when they are serialized again, while the additional context of their paragraphs is preserved.
- Removed the `lxml` dependency, which was used only by the DOCX converter.
- Removed the special handling of system messages for the retired OpenAI `o1-preview` and `o1-mini` models, for which the system message was left out of the request, with a warning. The system message is now sent to any model.
- Removed the warning issued when `parallel_tool_calls=True` was set for a model with the `openai/gpt-5` or `azure/gpt-5` name prefix. The warning was hardcoded for the first GPT-5 models, and was also issued for the later models with the same prefix.

### Fixed

- A `DocumentModelGroup` now also checks that its LLMs have unique roles when extraction starts, and raises `ValueError` before sending any requests if they do not. Previously, the roles were checked only when a group was created or its LLMs were reassigned, so changing the role of an LLM that was already in a group could leave several LLMs with the same role. The same aspects and concepts were then sent to each of these LLMs, and the extraction failed with an unrelated error after the requests were sent.
- Fallback LLMs now apply their own `num_retries_failed_request` and `max_retries_failed_request` during extraction. Previously, these were ignored for fallback LLMs, and the last fallback LLM failed the extraction on its first API error, e.g. a single rate limit error.
- A request that failed with an API error is no longer sent to the same LLM again during extraction. Previously, when a fallback LLM was configured or `raise_exception_on_extraction_error=False` was passed, such a request was repeated up to `max_retries_invalid_data` times without a delay, e.g. right after a rate limit error. `max_retries_invalid_data` now applies only when the LLM returns invalid data.
- With concurrency enabled, an API error that fails the extraction is now raised as `ModelAPIError` right away. Previously, the failed request was sent again first, and the error was raised as `ModelExtractionError` if `max_retries_invalid_data=0` was set.
- With concurrency enabled, requests to a fallback LLM are now paced by the `async_limiter` of that fallback LLM. Previously, the `async_limiter` set on a fallback LLM had no effect: requests to the fallback LLM shared the limiter of the primary LLM.
- With `auto_pricing=True`, the input tokens read from and written to the prompt cache of the LLM provider are now priced at the cache prices of the model. Previously, all input tokens were priced as regular input tokens, which overestimated the cost of the requests served from the cache, e.g. with the automatic prompt caching of OpenAI.
- With `auto_pricing=True`, each LLM request is now priced on its own. Previously, the tokens of several requests were summed up before being priced, e.g. when a document was processed in chunks or a chat response took several tool rounds. For the models priced higher above a certain number of input tokens per request (e.g. 200K), the sum could exceed that number even though no single request did, in which case all the tokens were priced at the higher prices.
- Token usage is now tracked for every LLM. Previously, `get_usage()` reported zero input and output tokens for an LLM that had neither `pricing_details` nor `auto_pricing` set, while the calls of that LLM were still recorded.
- An LLM can now be deserialized if its fallback LLMs differ from it, or from each other, only in `api_key` or `api_base`, e.g. the same model served from another endpoint. Previously, such an LLM could be created and serialized, but deserializing it failed with a validation error: as the API credentials are not included in the serialized data, the fallback LLM was taken for an exact copy of the primary LLM. A fallback LLM with the same configuration as the primary LLM or as another fallback LLM is still rejected when the fallback LLMs are set, i.e. when the primary LLM is created with them or `fallback_models` is assigned. This is no longer checked when another parameter of the primary LLM is assigned, so that the API credentials of a deserialized LLM and of its fallback LLMs can be set again in any order.
- An instance is now left unchanged when a value assigned to one of its parameters is rejected by a check that involves its other parameters. Previously, the error was raised but the rejected value was kept, i.e. the instance was left in the state that the error reported as invalid. This affected, among others, the role, output language, and fallback LLMs of a `DocumentLLM`, the output language of a `DocumentModelGroup`, the examples of a `JsonObjectConcept`, the sentences of a paragraph, and the sub-aspects of an aspect.

## [0.27.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.27.0) - 2026-08-13

### Added

- Added `Document.get_paragraph_index()` and `Document.get_sentence_index()` for locating paragraph and sentence objects within a document. Lookups are keyed by unique ID rather than text equality, so paragraphs/sentences with identical text (e.g. duplicate clauses in a contract) resolve to their specific occurrences. This makes the methods suitable for locating `reference_paragraphs`/`reference_sentences` returned during extraction, including after serialization round-trips, which preserve unique IDs. Lookups are O(1) on average via lazily built internal caches that are validated against the live document state and do not affect document equality, serialization, or cloning.

### Changed

- Upgraded pinned dependency versions: `litellm==1.96.2`, `openai==2.54.0`, `genai-prices==0.1.2`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

## [0.26.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.26.0) - 2026-07-28

### Added

- Added `"none"` and `"max"` reasoning effort values, extending the accepted set to the full vocabulary supported by litellm. `"none"` explicitly disables reasoning on models that support this value (e.g. newer gpt-5.x models, where omitting the parameter no longer disables reasoning), which differs from the default `None` (parameter not sent, provider default applies). `"max"` maps to the top effort level on Anthropic Claude models.

### Changed

- Model-specific support for `reasoning_effort` values is no longer validated by ContextGem: the hardcoded per-model checks (`"minimal"` only for gpt-5 models, `"xhigh"` only for gpt-5.2 models) were removed, as such hardcoded checks go stale with new model releases (e.g. gpt-5.3 and later gpt-5.x models, as well as newer Anthropic Claude models, also support `"xhigh"`). Validation is delegated to litellm / the provider API, which rejects unsupported model/effort combinations at request time. Configurations that previously raised `ValueError` at initialization now construct successfully and fail at request time if unsupported.
- Upgraded pinned dependency versions: `litellm==1.93.0`, `openai==2.49.0`, `genai-prices==0.0.72`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

### Fixed

- Fixed missing quotes in the aspect extraction prompt's JSON answer format. (PR [#96](https://github.com/shcherbak-ai/contextgem/pull/96))
- `top_p` now defaults to `None` (only `temperature` is sent) for compatibility with providers that reject requests setting both `temperature` and `top_p` (e.g. Claude 4.x models). (PR [#97](https://github.com/shcherbak-ai/contextgem/pull/97))

## [0.25.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.25.1) - 2026-06-06

### Added

- Added licenseal-based dependency license audit artifacts (`LICENSES.md`, `licenseal.review.toml`) and CI workflow.

### Fixed

- Fixed `Literal` type-hint serialization/deserialization so booleans, `None`, quoted strings, commas, and backslashes round-trip without being coerced or truncated.
- Fixed JSON Schema generation for mixed-type `Literal` enums and self-/mutually-referential class annotations, avoiding invalid `type` declarations and recursion failures.
- Fixed stale aspect nesting levels when reusing a sub-aspect as a document-level aspect via construction, assignment, `add_aspects()`, or pipeline assignment.
- Fixed deserialization paths that restored private attributes directly and could bypass property validation or validation that depends on restored private state.

## [0.25.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.25.0) - 2026-05-07

### Changed

- Concurrent fan-out for sub-aspect/sub-concept extraction and failed-chunk retries: when `use_concurrency=True`, these now run via `asyncio.gather` instead of awaiting per item, reducing wall-clock latency for documents whose aspects contain sub-aspect groups, and for retry batches when several chunks return invalid data at once.
- Upgraded pinned dependency version: `genai-prices==0.0.59`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

## [0.24.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.24.0) - 2026-05-05

### Added

- Runtime network egress scopes via [tethered](https://github.com/shcherbak-ai/tethered) on the two non-LLM paths ContextGem controls: SaT model downloads (HuggingFace) in `_load_sat_model`, and genai-prices pricing-data refresh (raw.githubusercontent.com) in `_calculate_auto_pricing_costs`. Each scope carries a `hint=` so that, if a host application's `tethered.activate()` blocks the call, the resulting `EgressBlocked` traceback names the host and the escape hatch (e.g. `paragraph_segmentation_mode='newlines'`, `auto_pricing_refresh=False`). LLM API calls are deliberately **not** scoped — endpoints are user-configured and remain governed by the host application's own policy.

### Changed

- Promoted `tethered` from a development-only dependency to a runtime dependency to support the production scopes above.

## [0.23.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.23.0) - 2026-05-04

### Changed

- Upgraded pinned dependency versions: `litellm==1.83.14`, `openai==2.24.0`, `genai-prices==0.0.57`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

### Docs

- Updated docs.

## [0.22.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.22.0) - 2026-03-16

### Added

- Network egress control in tests using [tethered](https://github.com/shcherbak-ai/tethered): enforces socket-level blocking of unauthorized outbound connections during VCR cassette replay, recording, and non-VCR tests.

### Changed

- Upgraded pinned dependency versions: `litellm==1.82.2`, `genai-prices==0.0.55`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

### Deprecated

- `DocxConverter` is deprecated and will be removed in v1.0.0. Use dedicated document conversion libraries (e.g., [Docling](https://github.com/docling-project/docling), [MarkItDown](https://github.com/microsoft/markitdown)) to convert files to text, then pass the result to `Document(raw_text=...)`.

### Removed

- Removed `llms.txt` generation (build script, pre-commit hook, and Sphinx integration). Use `AGENTS.md` and `CLAUDE.md` for AI assistant context instead.
- Removed `tests/url_security.py` cassette URL validation — replaced by tethered egress control.

## [0.21.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.21.0) - 2026-02-22

### Added

- Python 3.14 support.

### Fixed

- Fixed `Union.__getitem__()` incompatibility on Python 3.14, which caused `JsonObjectConcept` structure validation to fail when using complex union/optional type annotations (e.g., `Optional[Literal[...] | Literal[...]]`).
- Fixed incomplete `types.UnionType` detection in type normalization and validation, ensuring union types created with the `|` operator (Python 3.10+) are handled consistently across Python 3.10–3.14.

## [0.20.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.20.0) - 2026-02-22

### Added

- Auto-generate tool schemas from `@register_tool` decorated functions. Pass functions directly to `tools=[...]` — schemas are built automatically from type hints and docstrings. Explicit OpenAI-compatible schema dicts remain supported for full backward compatibility.
- Added `docstring-parser` dependency for extracting tool parameter descriptions from Sphinx/reST, Google, and NumPy style docstrings.

### Fixed

- Deterministic tool schema generation: `required` field ordering in auto-generated schemas is now sorted, preventing non-deterministic output from `frozenset` iteration across Python process invocations.
- Pinned `onnxruntime<1.24.0` for Python 3.10, as `onnxruntime` 1.24+ dropped Python 3.10 wheels. Python 3.11+ is unaffected and uses the latest version.

### Changed

- Upgraded pinned dependency versions: `litellm==1.81.14`, `openai==2.21.0`, `genai-prices==0.0.54`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

### Docs

- Added dedicated "Chat with Tools" documentation page with examples for auto-schema generation, supported type hints, `TypedDict` usage, custom schema overrides, and tool configuration options.
- Simplified quickstart tools example using the new `@register_tool` function-passing syntax.
- Updated `CONTRIBUTING.md` with AI assistant guidance and Fabric commands.

## [0.19.4](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.19.4) - 2025-12-19

### Fixed

- Applied fix for missing quote in JSON example format within prompt template. (PR [#86](https://github.com/shcherbak-ai/contextgem/pull/86))

### Added

- Added support for `reasoning_effort="xhigh"` for gpt-5.2 models.

### Changed

- Upgraded pinned dependency versions: `litellm==1.80.10`, `openai==2.13.0`, `genai-prices==0.0.49`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

## [0.19.3](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.19.3) - 2025-11-16

### Changed

- Upgraded pinned dependency versions: `litellm==1.80.0`, `openai==2.8.0`, `genai-prices==0.0.39`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases.

### Note

- The litellm 1.80.0 upgrade introduces Pydantic serialization warnings during async cleanup phases. These warnings are non-actionable and do not affect functionality. This is a known upstream issue tracked at [litellm PR #16299](https://github.com/BerriAI/litellm/pull/16299). ContextGem suppresses these warnings within its execution contexts; warnings may appear only during cleanup phases after operations complete.

## [0.19.2](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.19.2) - 2025-09-30

### Fixed

- Logging system refactored to use Python's standard library logging with namespaced logger (`contextgem`) for production-ready integration. Eliminates global state pollution, prevents conflicts with host application logging, and enables independent configuration. Replaced Loguru with colorlog for colored output.

## [0.19.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.19.1) - 2025-09-19

### Changed

- Enhanced documentation with pretty URLs (removing `.html` extensions) for improved SEO and user experience
- Added official documentation source notice to prevent confusion with unauthorized copies or mirrors

## [0.19.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.19.0) - 2025-09-08

### Added

- Tool calling support in `DocumentLLM.chat(...)`.

## [0.18.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.18.0) - 2025-09-01

### Added

- Chat: Added optional `chat_session` parameter (accepts a `ChatSession`) to preserve message history across turns in `DocumentLLM.chat(...)`. When this parameter is omitted, chat is single-turn, without message history.

## [0.17.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.17.1) - 2025-08-26

### Changed

- `DocxConverter`: Conversion speed improved by ~2X, significantly reducing processing time for DOCX files.

## [0.17.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.17.0) - 2025-08-24

### Added

- Multimodal LLM roles (`"extractor_multimodal"` and `"reasoner_multimodal"`) to support extraction of multimodal document-level concepts from both text and images. Previously, only text and vision roles were supported, requiring choosing either text or image context for extraction, not both.

## [0.16.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.16.1) - 2025-08-19

### Fixed

- Added support for `"minimal"` reasoning effort for gpt-5 models.

## [0.16.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.16.0) - 2025-08-19

### Added

- Reasoning-aware extraction prompts: Automatically enables private chain-of-thought guidance on models that support reasoning, yielding higher-quality outputs (no change for other models).

## [0.15.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.15.0) - 2025-08-14

### Added

- Auto-pricing for LLMs: enable via `auto_pricing=True` to automatically estimate costs using pydantic's `genai-prices`; optional `auto_pricing_refresh=True` refreshes cached price data at runtime.

### Refactor

- Public API made more consistent and stable: user-facing classes are now thin, well-documented facades over internal implementations. No behavior changes.
- Internal reorganization for maintainability and future-proofing.

### Docs

- Added guidance for configuring auto-pricing for LLMs.

## [0.14.4](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.14.4) - 2025-08-08

### Fixed

- Suppressed noisy LiteLLM proxy missing-dependency error logs (prompting to install `litellm[proxy]`) emitted by `litellm>=1.75.2` during LLM API calls. ContextGem does not require LiteLLM proxy features. Suppression is scoped to LiteLLM loggers.

## [0.14.3](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.14.3) - 2025-08-08

### Fixed

- Enabled `reasoning_effort` parameter for gpt-5 models by explicitly forwarding it via `allowed_openai_params`, since `litellm.get_supported_openai_params()` does not yet include this parameter for gpt-5 models.

## [0.14.2](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.14.2) - 2025-08-06

### Added

- Added warning for `gpt-oss` models used with `lm_studio/` provider due to performance issues (according to tests), with a recommendation to use Ollama as a working alternative (e.g., `ollama_chat/gpt-oss:20b`).

## [0.14.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.14.1) - 2025-08-06

### Added

- Added step-by-step usage guide in README, with brief descriptions of core components.
- Added new documentation on documents, extraction pipelines, and logging configuration.

### Changed

- Renamed `DocumentPipeline` to `ExtractionPipeline` to better reflect its purpose and scope. `DocumentPipeline` is maintained as a deprecated wrapper class for backwards compatibility until v1.0.0.
- Simplified logging config to use a single environment variable.

## [0.14.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.14.0) - 2025-08-02

### Added

- Added utility function `create_image()` for flexible image creation from various sources (file paths, PIL objects, file-like objects, raw bytes) with automatic MIME type detection.

### Changed

- Updated `image_to_base64()` utility function to accept more image source types (file-like objects and raw bytes) in addition to file paths.
- Made `temperature` and `top_p` parameters for DocumentLLM optional.

## [0.13.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.13.0) - 2025-07-30

### Changed

- Enhanced LLM prompts with XML tags for improved instruction clarity and higher-quality extraction outputs.
- Updated LabelConcept documentation with clearer distinction between multi-label and multi-class classification types.

### Fixed

- Fixed a bug where LabelConcept with multi-class classification type did not always return a label, as expected for multi-class classification tasks.

## [0.12.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.12.1) - 2025-07-27

### Added

- Explicit declaration of model vision capabilities: Added support for explicitly declaring vision capability when `litellm.supports_vision()` does not correctly identify a model's vision support. If a LLM is configured as a vision model and genuinely supports vision, but litellm fails to detect this capability, a warning will be issued. Users can manually set `_supports_vision=True` on the model instance to declare the capability and allow the model to accept image inputs.
- Warning for Ollama vision models: Added a warning prompting users to use the `ollama/` prefix instead of `ollama_chat/` for vision models, as `ollama_chat/` does not currently support image inputs.

### Changed

- Updated documentation to address vision capability detection issues and provide guidance on manual overrides.

## [0.12.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.12.0) - 2025-07-22

### Fixed

- BooleanConcept extraction for valid False values: Improved instructions in the concepts extraction prompt to fix a bug where no items were extracted for BooleanConcept with expected valid False values. The concept could be incorrectly considered "not addressed", resulting in empty extraction results.

### Changed

- Enhanced documentation: Added more details to parameter tables for Aspects API, Concepts API, and LLM config documentation.

## [0.11.1](https://github.com/shcherbak-ai/contextgem/releases/tag/v0.11.1) - 2025-07-11

### Fixed

- Allow disabling system message (e.g. for basic chat interactions): Added support for omitting system messages in DocumentLLM by allowing empty strings, which prevents sending any system message to the LLM. Introduced a warning in `llm.chat()/llm.chat_async()` when the default system message (optimized for extraction tasks) is used. Updated initialization to set default system message only when needed, ensuring flexibility for basic chat without a system message.

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
