# Changelog

All notable changes to ContextGem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), with the following additional categories:

- **Refactor**: Code reorganization that doesn't change functionality but improves structure or maintainability

## [1.0.0](https://github.com/shcherbak-ai/contextgem/releases/tag/v1.0.0) - 2026-10-06

### Added

- Added the `modality` parameter to concepts, which sets the document content a concept is extracted from: `"text"` (default), `"image"`, or `"multimodal"` (text and/or images, whichever the document has). `"image"` and `"multimodal"` are supported for document-level concepts only. Previously, the content was determined by the role of the LLM (see "Changed").
- Added the `fallback_models` parameter to `DocumentLLM`, which accepts a list of fallback LLMs and replaces the `fallback_llm` parameter, which accepted a single one (see "Removed"). Fallback LLMs are tried in the given order until one of them succeeds, for both extraction and chat, e.g. to fall back across several providers during an outage. Each fallback LLM must have `is_fallback=True` and the same role, output language, system message, and tool configuration as the primary LLM, must differ in configuration from the primary LLM and from the other fallback LLMs, and cannot have fallback LLMs of its own. Usage and cost are tracked for the primary LLM and each of its fallback LLMs.
- Added tracking of cached input tokens. The usage returned by `get_usage()` now includes `cache_read` and `cache_write`: the number of input tokens that the LLM provider reports as read from, and written to, its prompt cache. Both are a part of the `input` token count, and are zero if the provider does not report such tokens.
- Added the optional `cache_read_per_1m_tokens` and `cache_write_per_1m_tokens` parameters to `ModelPricing` (previously `LLMPricing`, see "Changed"), which set the prices of the input tokens read from and written to the prompt cache of the LLM provider. If a cache price is not set, such tokens are priced as regular input tokens, as before. Serialized `ModelPricing` data now includes these keys.
- Added support for `Annotated` type hints in the functions used as tools in chat. A string in `Annotated`, or pydantic's `Field(description=...)`, describes a parameter and takes precedence over the description in the docstring. The constraints set with `Field` (`gt`, `ge`, `lt`, `le`, `multiple_of` for numbers; `min_length`, `max_length` for strings and collections; `pattern` for strings) are added to the tool schema sent to the LLM, and are enforced when the LLM calls the tool: for an invalid value, the tool is not called, and the LLM receives the validation error. `Annotated` is also supported for the items of collections and the fields of a `TypedDict`.
- Added `DocumentDecisionModel`, a class for decision models, next to `DocumentLLM`. A decision model takes a text and typed questions (yes/no, pick one option, rate on a scale) and returns the probabilities of the answers; it generates no text. A `DocumentDecisionModel` extracts `BooleanConcept`, `LabelConcept`, and `RatingConcept`, from a document or from an aspect: the questions of the concepts extracted in one call are sent in one request, with the text of the document or of the paragraphs of the aspect. It also extracts aspects, including sub-aspects: each paragraph is sent in a request of its own, with one yes/no question per aspect, and becomes an extracted item of each aspect that it belongs to with a probability of the decision threshold (0.5 by default) or more. For an aspect with `reference_depth="sentences"`, the sentences of the paragraphs that belong to the aspect are then asked about in the same way, and the sentences that belong to the aspect become its extracted items (see below). It has the extraction methods of `DocumentLLM` and no `chat()`, and can be used on its own, in a `DocumentModelGroup`, and in fallback chains together with LLMs (see "Changed"). Requests are sent through the Decisions API of `litellm`, which serves OpenAI, TypeSafe, Perplexity, OpenRouter, Cloudflare, and local or self-hosted servers such as Ollama and LM Studio (under the `strands_decider/` prefix, with the address of the server as `api_base`). `ValueError` is raised before any requests are sent for what a decision model cannot extract: concepts with open values (`StringConcept`, `NumericalConcept`, `DateConcept`, `JsonObjectConcept`), concepts with the `"image"` or `"multimodal"` modality, justifications, references for concepts, and concepts with more labels or rating levels than `litellm` accepts in a question. A model of a provider that `litellm` serves no decision models of is rejected when the decision model is created. Failed requests are retried as for LLMs. A response that cannot be used, e.g. with a refused answer, is not requested again: the next fallback model is tried, and `ModelExtractionError` is raised if there is no fallback model left to try. Usage and cost are tracked as for LLMs. `auto_pricing` works for the models whose prices of decision requests `genai-prices` has (the TypeSafe models), and is rejected for a decision model for which `genai-prices` has the prices of a text-generating model, i.e. a price of output tokens (e.g. `gpt-6-luna`, for which it has only the chat prices, whichever provider serves it), and for self-hosted ones (`strands_decider/`): set `pricing_details` to get their costs calculated. See the [documentation](https://contextgem.dev/decision_models/decision_models/).
- Added `decision_threshold`, the lowest probability of "yes" with which a decision model takes a yes/no question as answered "yes": a paragraph or a sentence belongs to an aspect, a `BooleanConcept` is `True`, a label of a multi-label `LabelConcept` is selected. It is a parameter of `DocumentDecisionModel` (0.5 by default), which applies to everything that the model extracts, and of `Aspect`, `BooleanConcept` and `LabelConcept` (not set by default), where it applies instead of the threshold of the decision model. It must be above 0 and not above 1. It does not apply to single-label and rating concepts, for which the most probable option is taken, and cannot be set for a `LabelConcept` with `classification_type="multi_class"`. A sub-aspect does not get the threshold of its aspect. It does not change the `probabilities` of the extracted items, and is ignored when an LLM extracts the aspect or the concept. The serialized data of aspects, `BooleanConcept` and `LabelConcept` now includes the `decision_threshold` key (`null` if not set); data serialized by earlier versions has no such key, which needs no update.
- `max_paragraphs_to_analyze_per_call` applies to the concepts that a `DocumentDecisionModel` extracts. When it is set and the text has more paragraphs, the paragraphs are split into consecutive parts of at most that many paragraphs, as for an LLM, and one request is sent for each part, so that a long document can be used with a model that has a small context, e.g. a local one. Each concept gets one extracted item for each part, in the order of the parts, each with its own `probabilities`: the answers are not merged across the parts, i.e. combining them is up to the calling code, and a multi-label `LabelConcept` gets no item for a part in which no label reaches the decision threshold. With `use_concurrency=True`, the requests for the parts are sent concurrently, at the rate of the `async_limiter` of the model. Without the parameter, the text is sent as a whole in one request. For aspects, the parameter changes nothing, as each paragraph is sent in a request of its own.
- A concept with `singular_occurrence=True` cannot be extracted by a `DocumentDecisionModel` in parts: with `max_paragraphs_to_analyze_per_call` set, `ValueError` is raised before any requests are sent, also when the decision model is a fallback of the model that the concept is routed to. A decision model answers on every part of a text, so it cannot stop at the first part that gives an item, as an LLM does. Without `max_paragraphs_to_analyze_per_call`, such a concept is extracted, with one item at most.
- When a `DocumentDecisionModel` extracts aspects, including sub-aspects, a paragraph that has `additional_context` is sent with it, as an LLM is given it: the text of the request has the paragraph and its additional context, each with a label, and the question says that the paragraph is what is to be judged. A paragraph without additional context is sent as its text alone. Concepts are not affected.
- A `DocumentDecisionModel` extracts aspects with `reference_depth="sentences"`, including sub-aspects, in two stages. First, each paragraph is sent in a request of its own, as for `reference_depth="paragraphs"`. Then, each sentence of each paragraph that belongs to the aspect is sent in a request of its own, with the sentence, the paragraph that it is from as its context, and the `additional_context` of the sentence and of the paragraph, if set. A sentence with a probability of the decision threshold or more becomes an extracted item, with the sentence in `reference_sentences`, its paragraph in `reference_paragraphs`, and the `probabilities` of the sentence: the same threshold applies at both stages. A paragraph that belongs to the aspect while none of its sentences does gives no items. The number of requests is the number of paragraphs plus the number of sentences in the paragraphs that belong to the aspect.
- Added the `probabilities` attribute to extracted items: a dictionary that maps each possible answer to its probability, as returned by a decision model. The keys are the answers as they appear as item values: `True` and `False` for an item of a `BooleanConcept` and for an item of an aspect (whether the paragraph, or the sentence for an aspect with `reference_depth="sentences"`, belongs to the aspect), the labels for an item of a `LabelConcept`, and the integers of the rating scale for an item of a `RatingConcept`. For a multi-label `LabelConcept`, each label has the probability of its own yes/no question, so the probabilities do not sum to 1. `probabilities` is `None` for every item extracted by an LLM. Two items that differ only in their probabilities are not equal.

### Changed

- **Breaking:** Renamed the classes, parameters, and attributes that are not specific to LLMs, so that their names refer to models in general. `DocumentLLM` and `LLMToolLoopLimitError` keep their names. The old names are removed, and using them raises an error. See the [migration guide](https://contextgem.dev/migration_v1/) for updating the code and the serialized data.
  - `DocumentLLMGroup` is renamed to `DocumentModelGroup`, and its `llms` parameter and attribute to `models`.
  - The `llm_role` parameter and attribute of aspects and concepts is renamed to `model_role`, as is the `llm_role` argument of `get_usage()`, `get_cost()`, and `reset_usage_and_cost()` of `DocumentModelGroup`.
  - The `llm_roles` property of documents, pipelines, and aspects is renamed to `model_roles`.
  - `LLMPricing` is renamed to `ModelPricing`.
  - `LLMAPIError` is renamed to `ModelAPIError`, and `LLMExtractionError` to `ModelExtractionError`.
- **Breaking:** Serialized data uses the new names: the `model_role` key of aspects and concepts, the `models` key and the `DocumentModelGroup` class name of model groups, the `ModelPricing` class name of pricing details, and the class names of the usage and cost containers stored in serialized LLMs (always empty, as the usage and cost statistics are not serialized). Aspects, concepts, LLMs, and model groups serialized by earlier versions, as well as the documents and pipelines that have aspects or concepts, can no longer be deserialized as is. Update the serialized data as described in the [migration guide](https://contextgem.dev/migration_v1/).
- **Breaking:** Model roles are now free-form labels. The `role` parameter of `DocumentLLM` and the `model_role` parameter of aspects and concepts accept any label of up to 64 printable characters (the surrounding whitespace is removed; line breaks, tabs, and other control or invisible characters are not allowed) and default to `None` (no role), instead of one of the six predefined role names (`"extractor_text"`, `"reasoner_text"`, `"extractor_vision"`, `"reasoner_vision"`, `"extractor_multimodal"`, `"reasoner_multimodal"`; for aspects, one of the two `*_text` names), with `"extractor_text"` as the default. A role only routes aspects and concepts to the LLMs of a `DocumentModelGroup`: an aspect or concept is extracted by the LLM with the `role` equal to its `model_role`, and an aspect or concept without `model_role` is extracted by the LLM without `role`. A role no longer implies the capabilities of the model or the content it can process, so the same model no longer needs to be added to a group several times, with different roles, to process both text and images. Each LLM in a group must still have a unique role, i.e. only one LLM in a group can have no role. The predefined role names have no special meaning anymore: if still used, they are treated as any other role labels, and no longer determine the content a concept is extracted from. See the [migration guide](https://contextgem.dev/migration_v1/) for updating the code and the serialized data.
- **Breaking:** A `DocumentLLM` used on its own (not in a group) now extracts all the aspects and concepts it is given, regardless of their `model_role`. Previously, it skipped the aspects and concepts with a `model_role` different from its `role`, with a warning. To extract only some of the aspects or concepts with a single LLM, use the `from_aspects` / `from_concepts` parameters of the extraction methods.
- **Breaking:** A `DocumentModelGroup` now raises `ValueError` if an aspect or concept to be extracted has a `model_role` that no LLM in the group has. This is checked before sending any requests for that extraction method, including for the nested aspects and concepts that the method processes. Previously, such aspects and concepts were skipped: with a warning in `extract_all()`, and silently in the other extraction methods.
- **Breaking:** The content a concept is extracted from is now determined by the `modality` of the concept instead of the role of the LLM. Concepts with different modalities can be extracted by the same LLM, in separate calls.
- Vision and reasoning capabilities of a model are no longer checked against the role of the LLM when the LLM is created, i.e. the related warnings are no longer issued. Instead, `ValueError` is raised before any requests are sent, if the LLM that is to extract concepts from images, or any of its fallback LLMs, is not detected as vision-capable (unless `_supports_vision=True` is set on the LLM). The same applies to the fallback LLMs when images are passed to `chat()`. Previously, such error was raised only when the images were sent to the model, i.e. for a fallback LLM only after the primary LLM failed. As before, the reasoning-specific prompt instructions are used based on the detected reasoning support of the model. The warning about the `ollama_chat/` prefix not supporting image inputs is now issued when images are sent to such a model, instead of when the LLM is created.
- **Breaking:** The `model_role` argument of `get_usage()`, `get_cost()`, and `reset_usage_and_cost()` of `DocumentModelGroup` is now validated in the same way as the roles assigned to LLMs: the surrounding whitespace is removed, and an invalid role, e.g. an empty string, raises `ValueError`. Previously, the argument was compared as passed, and an empty string selected all LLMs. The LLM without a role cannot be selected with `model_role`, as `None` selects all LLMs: filter the returned entries by `role is None`, or call the method on that LLM. Previously, it was selected with `"extractor_text"`.
- **Breaking:** The `model_roles` property of documents, pipelines, and aspects now also includes the roles assigned to sub-aspects and their concepts, and contains `None` if there are aspects or concepts without a role, where it contained `"extractor_text"` before. Code that expects a set of strings, e.g. sorts the roles, has to handle `None`.
- **Breaking:** The `list_roles` property of `DocumentLLM` and `DocumentModelGroup`, and the `role` of the entries returned by `get_usage()` and `get_cost()`, are now `None` for an LLM without a role. Previously, they were `"extractor_text"` for an LLM created without a role.
- Serialized concept data now includes the `modality` key. Documents and pipelines serialized by earlier versions store the predefined role names and no modality, and need to be updated as described in the [migration guide](https://contextgem.dev/migration_v1/).
- **Breaking:** Serialized `DocumentLLM` data now stores fallback LLMs under the `fallback_models` key instead of the `fallback_llm` key. An LLM or model group serialized by earlier versions has the `fallback_llm` key in each LLM, even if no fallback LLM was assigned, and can no longer be deserialized as is. Update the serialized data as described in the [migration guide](https://contextgem.dev/migration_v1/), or create the LLMs anew.
- Failed LLM API requests are now retried by ContextGem instead of litellm, with `num_retries_failed_request` keeping its meaning. Only transient errors (request timeout, rate limit, server or connection error) are retried; non-transient errors (e.g. invalid request, failed authentication, exceeded context window) fail right away or are passed to a fallback LLM, instead of being retried as before. Retries are performed with exponential backoff and jitter (about 1, 2, 4, 8 seconds, capped at 10 seconds, each randomly shortened by up to a half); previously, only rate limit errors were retried with a backoff, without jitter, and other errors were retried immediately.
- `ModelAPIError.retry_count`, as well as the corresponding warning messages, now report the number of retries actually attempted for the failed request, rather than the configured number of retries.
- The warning issued when the pricing data of a model is not found with `auto_pricing=True` now also suggests setting `auto_pricing_refresh=True`, unless it is already set, as the pricing data bundled with `genai-prices` may not include a recently released model.
- Upgraded pinned dependency versions: `litellm==1.104.2`, `genai-prices==0.1.9`. Versions remain pinned to maintain stability and avoid occasional breaking changes and API inconsistencies observed in previous unpinned releases. `openai` remains pinned to `2.54.0`, the latest version supported by `litellm`. As of this version, `litellm` installs `boto3` as a required dependency.
- **Breaking:** A tool for chat is now defined in a single way: as a function decorated with `@register_tool`, which is passed to `tools` of `DocumentLLM`. `tools` no longer accepts tool definitions as dictionaries in the OpenAI-compatible schema, and `llm.tools` returns the functions instead of the tool schemas. The schema of a tool is always generated from the function. To update the tools defined with dictionaries, see the [migration guide](https://contextgem.dev/migration_v1/).
- **Breaking:** `@register_tool` now raises an error for a function that has no docstring, or that accepts `*args` or `**kwargs`. Previously, such a function was accepted: a tool without a docstring was sent to the LLM without a description, and `*args` and `**kwargs` were ignored. A parameter without a type hint is now reported when the function is decorated, rather than when the LLM is created.
- Functions decorated with `@register_tool` can now be assigned to the `tools` of an existing `DocumentLLM` (`llm.tools = [my_tool]`), as a list or a tuple. Previously, functions were accepted only when the LLM was created.
- **Breaking:** Tools with the same name are now rejected when they are set on an LLM. Previously, the last of them silently replaced the others.
- **Breaking:** A fallback LLM must now have the same `system_message` as the primary LLM, which is checked when the fallback LLMs are set and when the system message of the primary LLM is changed. If the primary LLM is created with `system_message`, e.g. with an empty string to disable it for a chat, pass the same value to each of its fallback LLMs. Previously, a fallback LLM could have another system message: a fallback LLM created without `system_message` used the default system message, which is written for extraction, also in a chat for which the primary LLM had another system message or none. In a chat with a `ChatSession`, a fallback LLM was sent the system message of the session instead of its own, or, if the session had none, added its own system message to the session, which was then sent with the later requests of the primary LLM as well. See the [migration guide](https://contextgem.dev/migration_v1/).
- A prompt that is found to exceed the context window of an LLM before the request is sent is now handled as any other non-transient failure of that LLM: the next fallback LLM, which may have a larger context window, is tried, and `ValueError` is raised only if there is no fallback LLM left to try. Previously, `ValueError` was raised right away, and the fallback LLMs were not tried. This applies to both extraction and chat.
- In chat, the request parameters of the fallback LLMs are now checked before any request is sent: `ValueError` is raised if `parallel_tool_calls` is set for a fallback LLM that is not detected as supporting parallel tool calls, or if the max output tokens of a fallback LLM exceed the limit of its model. Previously, this was checked only when the fallback LLM was to be used, i.e. after the primary LLM failed.
- **Breaking:** In chat, an empty response of an LLM is now handled as a failure of that LLM: the next fallback LLM is tried, and `ModelAPIError` is raised if there is no fallback LLM left to try. Previously, a response without a text was passed to the fallback LLM or raised `RuntimeError`, while a response with an empty text was returned as the answer, or failed with a validation error when a chat session was used.
- A fallback LLM can now have the same tools as the primary LLM in a different order. Previously, it was rejected as having a different tool configuration.
- **Breaking:** A function used as a tool in chat must return a non-empty string: for an empty string, `ValueError` is now raised from `chat()`. Previously, an empty string failed the request with a validation error reported as `ModelAPIError`.
- **Breaking:** The output tokens of a response are now limited by a single parameter of `DocumentLLM`, `max_tokens`, which applies to all models and is not set by default: no limit is sent with the requests, and the default of the model applies, which for most providers is the maximum output of the model. Previously, `max_tokens` (4096 by default) applied to the models without reasoning support, and `max_completion_tokens` (16000 by default, see "Removed") to reasoning (CoT-capable) models, while the other of the two parameters was ignored. A `max_tokens` value set for a reasoning model, which was ignored before, now limits the output of the model, including its reasoning tokens. `max_tokens` is checked against the maximum output of the model only if it is set, so the default no longer raises `ValueError` for the models with a maximum output below the previous default limits. See the [migration guide](https://contextgem.dev/migration_v1/).
- **Breaking:** A response cut off at the max output tokens is no longer requested again during extraction, as the same request would be cut off in the same way: the next fallback LLM is tried, and `ModelAPIError` is raised if there is no fallback LLM left to try, with a message telling that the response was cut off and how to get a complete one (or a warning is issued, if `raise_exception_on_extraction_error=False` is passed). Previously, such a response was handled as invalid data: it was requested again up to `max_retries_invalid_data` times, and the extraction failed with `ModelExtractionError` reporting invalid JSON.
- In chat, a response cut off at the max output tokens is now returned with a warning. If such a response has no text, the `ModelAPIError` raised for the empty response tells that the response was cut off. Previously, a cut-off response was returned without a warning.
- **Breaking:** The `temperature` of `DocumentLLM` is no longer set by default: the parameter is not sent, and the default of the model applies. Previously, it defaulted to 0.3, which was sent to the models without reasoning support only. To keep the previous behavior for such a model, e.g. a local model, set `temperature=0.3`. See the [migration guide](https://contextgem.dev/migration_v1/).
- `temperature` and `top_p` are now sent to any model if they are set, including reasoning (CoT-capable) models, for which they were ignored before. Whether a value is supported for the model is left to `litellm` and the LLM provider: if `litellm` reports a parameter as not supported for the model (e.g. a temperature for OpenAI's GPT-5 and GPT-6 models, which accept only their default one), the request is sent without it. The message logged in this case is now a warning instead of an error.
- The default `timeout` of `DocumentLLM` is raised from 120 to 600 seconds. Responses are not streamed, so a response has to be received in full within the timeout, and a long response of a reasoning model may take more than 120 seconds. A request that times out is retried as a transient error, so a request to an unresponsive endpoint now takes longer to fail, unless a lower `timeout` is set.
- The default `async_limiter` of `DocumentLLM`, which paces the requests sent with `use_concurrency=True`, now allows 10 requests per 10 seconds instead of 3. Set the limiter explicitly if this exceeds the rate limits of your LLM provider account.
- **Breaking:** `DocumentModelGroup` is moved from the `contextgem.public.llms` module to the new `contextgem.public.models` module, as a model group is not specific to LLMs. It can no longer be imported from `contextgem.public.llms`. Importing it from `contextgem` works as before. Serialized data is not affected, as it stores the class names without the module paths. See the [migration guide](https://contextgem.dev/migration_v1/).
- The `models` of a `DocumentModelGroup` and the `fallback_models` of a `DocumentLLM` now accept decision models (`DocumentDecisionModel`) as well as LLMs. In a group, aspects and concepts are routed to a decision model by its role in the same way as to an LLM, and the `output_language` of the group applies to its LLMs only. An LLM can have a decision model as a fallback only if the decision model can extract everything that is given or routed to the LLM, which is checked when an extraction method is called, before any requests are sent. A decision model can always have an LLM as a fallback. The requirement that a fallback LLM has the same output language, system message, and tool configuration as the primary LLM applies between two LLMs. In `chat()`, a decision model among the fallback models of an LLM is skipped.
- Serialized data of extracted items now includes the `probabilities` key: a list of `[answer, probability]` pairs, or `null` for an item without probabilities. Data serialized by earlier versions has no such key, which needs no update: its items are loaded with `probabilities=None`. Data serialized by this version cannot be loaded by earlier versions.
- The parameters that `DocumentLLM` has in common with other kinds of models (`model`, `api_key`, `api_base`, `role`, `num_retries_failed_request`, `timeout`, the pricing and the fallback parameters) now come first in the `repr()` of an LLM and in the key order of its serialized data. Serialized data is loaded regardless of the order of its keys.
- A fallback LLM is now compared with the primary LLM, and with the other fallback LLMs, by its class as well as by its configuration parameters: an instance of a subclass of `DocumentLLM` with the same parameters as the primary LLM is no longer rejected as its exact copy.

### Removed

See the [migration guide](https://contextgem.dev/migration_v1/) for updating the code and the serialized data affected by the removals.

- **Breaking:** Removed the `fallback_llm` parameter and attribute of `DocumentLLM`. Use `fallback_models` instead, e.g. `fallback_models=[fallback_llm]`. Passing `fallback_llm` to `DocumentLLM`, or reading or assigning the attribute, now raises an error.
- **Breaking:** Removed `DocumentPipeline`, deprecated since v0.14.1. Use `ExtractionPipeline` instead, which has the same parameters and methods. A pipeline serialized as `DocumentPipeline` can no longer be deserialized as is.
- **Breaking:** Removed `RatingScale`, deprecated since v0.10.0. Pass the rating scale of a `RatingConcept` as a tuple of `(start, end)` integers instead, e.g. `rating_scale=(1, 5)`. Documents, pipelines, aspects, and concepts serialized with a rating scale defined as a `RatingScale` object can no longer be deserialized as is.
- **Breaking:** Removed `DocxConverter`, deprecated since v0.22.0, together with the DOCX converter exceptions (`DocxConverterError` and its subclasses). Use a dedicated document conversion library (e.g. [Docling](https://github.com/docling-project/docling), [MarkItDown](https://github.com/microsoft/markitdown)) to convert files to text, then pass the result to `Document(raw_text=...)`.
- **Breaking:** Removed the Markdown text that `DocxConverter` attached to documents and paragraphs and that was sent to the LLM instead of the plain text. Text is now always extracted from `raw_text`. Documents converted from DOCX and serialized by earlier versions need no update for this removal: when they are deserialized, their Markdown text is ignored, and it is not kept when they are serialized again, while the additional context of their paragraphs is preserved.
- Removed the `lxml` dependency, which was used only by the DOCX converter.
- Removed the special handling of system messages for the retired OpenAI `o1-preview` and `o1-mini` models, for which the system message was left out of the request, with a warning. The system message is now sent to any model.
- Removed the warning issued when `parallel_tool_calls=True` was set for a model with the `openai/gpt-5` or `azure/gpt-5` name prefix. The warning was hardcoded for the first GPT-5 models, and was also issued for the later models with the same prefix.
- **Breaking:** Removed the `max_completion_tokens` parameter and attribute of `DocumentLLM`. Use `max_tokens` instead, which now applies to all models (see "Changed"). Passing `max_completion_tokens` to `DocumentLLM`, or reading or assigning the attribute, now raises an error. An LLM or model group serialized by earlier versions has the `max_completion_tokens` key in each LLM, and can no longer be deserialized as is.

### Fixed

- A `DocumentModelGroup` now also checks that its LLMs have unique roles when extraction starts, and raises `ValueError` before sending any requests if they do not. Previously, the roles were checked only when a group was created or its LLMs were reassigned, so changing the role of an LLM that was already in a group could leave several LLMs with the same role. The same aspects and concepts were then routed to each of these LLMs, and the extraction failed with an unrelated error (that they are already processed) after requests were sent.
- Fallback LLMs now apply their own `num_retries_failed_request` and `max_retries_failed_request` during extraction. Previously, these were ignored for the fallback LLM, which failed the extraction on its first API error, e.g. a single rate limit error.
- A request that is sent again because the LLM returned invalid data (`max_retries_invalid_data`) is now retried on transient API errors as any other request, up to `num_retries_failed_request` times. Previously, such a request was sent without retries, so a single transient error on it, e.g. a rate limit error, failed the extraction with an API error.
- A request that failed with an API error is no longer sent to the same LLM again during extraction. Previously, when a fallback LLM was configured or `raise_exception_on_extraction_error=False` was passed, such a request was repeated up to `max_retries_invalid_data` times without a delay, e.g. right after a rate limit error. `max_retries_invalid_data` now applies only when the LLM returns invalid data.
- With concurrency enabled, an API error that fails the extraction is now raised as `ModelAPIError` right away. Previously, the failed request was sent again first, and the error was raised as `ModelExtractionError` if `max_retries_invalid_data=0` was set.
- With `raise_exception_on_extraction_error=False`, an LLM that has `max_retries_invalid_data=0` and no fallback LLMs now issues a single warning for a call that fails because its API request failed, or because its response was cut off at the max output tokens: the warning about that failure. The warning that the LLM returned invalid JSON, which is wrong for such a call, is no longer issued for it.
- With concurrency enabled, requests to a fallback LLM are now paced by the `async_limiter` of that fallback LLM. Previously, the `async_limiter` set on a fallback LLM had no effect: requests to the fallback LLM shared the limiter of the primary LLM.
- With concurrency enabled, all the requests to a model that are sent in the same event loop are now paced by the same limiter, at the rate of the `async_limiter` of the model. Previously, each extraction call had a limiter of its own, so the calls that run concurrently, e.g. for the sub-aspects and for the concepts of different aspects in `extract_all()`, each sent requests at the full rate, and together exceeded the rate several times. The same applied to the extractions run concurrently with the async methods, e.g. from several documents. An extraction with concurrency may take longer now: set `async_limiter` to the rate limits of your provider account.
- With `auto_pricing=True`, the input tokens read from and written to the prompt cache of the LLM provider are now priced at the cache prices of the model. Previously, all input tokens were priced as regular input tokens, which overestimated the cost of the requests served from the cache, e.g. with the automatic prompt caching of OpenAI.
- With `auto_pricing=True`, each LLM request is now priced on its own. Previously, the tokens of several requests were summed up before being priced, e.g. when a document was processed in chunks or a chat response took several tool rounds. For the models priced higher above a certain number of input tokens per request (e.g. 200K), the sum could exceed that number even though no single request did, in which case all the tokens were priced at the higher prices.
- Token usage is now tracked for every LLM. Previously, `get_usage()` reported zero input and output tokens for an LLM that had neither `pricing_details` nor `auto_pricing` set, while the calls of that LLM were still recorded.
- An LLM can now be deserialized if its fallback LLMs differ from it, or from each other, only in `api_key` or `api_base`, e.g. the same model served from another endpoint. Previously, such an LLM could be created and serialized, but deserializing it failed with a validation error: as the API credentials are not included in the serialized data, the fallback LLM was taken for an exact copy of the primary LLM. A fallback LLM with the same configuration as the primary LLM or as another fallback LLM is still rejected when the fallback LLMs are set, i.e. when the primary LLM is created with them or `fallback_models` is assigned. This is no longer checked when another parameter of the primary LLM is assigned, so that the API credentials of a deserialized LLM and of its fallback LLMs can be set again in any order.
- An instance is now left unchanged when a value assigned to one of its parameters is rejected by a check that involves its other parameters. Previously, the error was raised but the rejected value was kept, i.e. the instance was left in the state that the error reported as invalid. This affected, among others, the role, output language, and fallback LLMs of a `DocumentLLM`, the output language of a `DocumentModelGroup`, the examples of a `JsonObjectConcept`, the sentences of a paragraph, and the sub-aspects of an aspect.
- In chat with tools, tool calls that the LLM returns together with a text in the same response are now executed for the models that litellm calls through the OpenAI Responses API, e.g. OpenAI reasoning models (`gpt-5.4` and later) that are given function tools. Previously, such tool calls were ignored, and `chat()` returned the text that came with them as the final answer, without calling the remaining tools.
- In chat with tools, an assistant turn in which the LLM requests tool calls without returning a text is now sent back to the LLM without a text. Previously, the placeholder text `<tool_call_response>` was sent as the text of such turns. Some LLMs repeated it in their responses, and with some chat templates (e.g. Mistral models on Ollama) it replaced the tool calls in the conversation rendered for the LLM, so that the LLM stopped calling the remaining tools. The `content` of a chat message is therefore optional: it is `None` only for such assistant turns, which are stored separately from `ChatSession.messages`.
- The function passed to `tools` is now the one that is called when the LLM uses the tool. Previously, the function was looked up by its name in the calling code, so a tool passed as a module attribute (`tools=[my_tools.search]`) or imported under another name was rejected with "No registered handler found".
- In the schema of a tool, a `TypedDict` field marked as `NotRequired` is no longer listed as required when the annotations of the module are evaluated lazily (`from __future__ import annotations`).
- The documentation no longer describes passing a custom tool schema to `@register_tool(schema=...)`: this was not supported, and raised `TypeError`.
- A tool whose type hints cannot be resolved, e.g. because they use a type defined inside a function in a module with `from __future__ import annotations`, is now reported with an error that names the type that was not found and how to fix it. Previously, the error stated that a supported type hint, e.g. `list[...]`, was not supported.
- The tools of an LLM are no longer used during extraction. Previously, the tools, together with `tool_choice` and `parallel_tool_calls`, were sent with the requests of an extraction, and the tool calls returned by the LLM were executed, although tools are documented, and warned about, as ignored by the extraction methods.
- In chat with tools, an error in a tool function is no longer handled as an error of the LLM API. A tool that returns a value other than a string now raises `TypeError` from `chat()` right away. Previously, `ModelAPIError` was raised instead ("Exception occurred while calling LLM API"), and if fallback LLMs were configured, each of them was tried first, calling the tool again.
- In chat with tools, when the LLM calls a tool that does not exist, it now receives an error with the names of the available tools, and can correct the call. Previously, such a call failed the request with `ModelAPIError`, or passed the request to a fallback LLM.
- In chat with tools, no tool is called when a response of the LLM has a tool call that cannot be answered, i.e. a tool call without the name of the tool or without the ID of the call. Previously, the tools of the other tool calls in that response, and the tool of a call without an ID, were called before the request failed.
- The usage and cost of the requests already sent are now recorded when a chat with tools, or an extraction that takes several requests (e.g. with `max_paragraphs_to_analyze_per_call` set), ends with an error: when `tool_max_rounds` is reached, when a request fails after other requests succeeded, or when a chat is cancelled, e.g. on a timeout set by the calling code. Previously, the tokens used by such requests were not counted. The request that failed is now recorded as a call without a response also when the error is raised, rather than only when a warning is issued instead or a fallback LLM is tried. A request that failed after a round of tool calls no longer records the first request of the chat twice.
- A `seed` of `0` set for `DocumentLLM` is now sent with the requests. Previously, it was left out, as if no seed was set.
- An aspect that is extracted again with `overwrite_existing=True` now has the reference paragraphs and sentences of its newly extracted items only. Previously, the references of the earlier extraction were kept, so the sub-aspects and the concepts of the aspect that were extracted afterwards were also given the paragraphs that were no longer extracted for the aspect.
- An aspect or a concept that is passed to an extraction method more than once, e.g. `from_concepts=[concept, concept]`, is now extracted once, and is returned once. Previously, without concurrency, it was listed in the prompt as many times as it was passed, and got the items of each listing, i.e. the same items several times. With concurrency, a request was sent for each time it was passed, and the extraction then failed with a `ValueError` on the aspect or the concept being already processed. The sub-aspects of an aspect passed more than once were extracted for each time as well.
- With concurrency enabled, when the extraction with one model of a `DocumentModelGroup` fails, the extraction with the other models of the group is now stopped before the error is raised: their requests in flight are cancelled, and no requests are sent afterwards. Previously, the other models kept sending requests, and updating the aspects and concepts, in the background after the calling code received the error. The same applies to whatever else is run concurrently, e.g. the extraction of the sub-aspects of several aspects, and the tools of a chat called in parallel: when a tool fails the chat by returning a value other than a string, the tools still running are cancelled. The usage of the requests completed by then is recorded.
- The `api_key` of an LLM is no longer shown when the LLM is printed (`repr()`, `str()`), on its own or as a part of another object, e.g. in logs, or when what the `errors()` method of a validation error returns is printed, for an input that includes the LLM, such as the `fallback_models` of another LLM or the `models` of a `DocumentModelGroup`. Previously, the key was shown there in clear text.
- The text of a validation error of a `DocumentLLM`, a `DocumentDecisionModel` or a `DocumentModelGroup` no longer shows the values that failed the validation (`input_value=..., input_type=...`), so that no part of an `api_key` can appear there, e.g. in logs. The error still names the invalid parameter and the reason, and the values are still returned by the `errors()` and `json()` methods of the error: `json()` writes them out in full, including the `api_key` of an LLM that is among them. Previously, an `api_key` that was rejected itself was shown, e.g. one of a wrong type, or one passed to a `DocumentModelGroup`, which has no such parameter. The beginning or the end of the key passed as the first or the last parameter of a model was shown as well, when a check that involves several parameters failed.
- A tab, a line break or another whitespace character between two words is now replaced with a space when a text is prepared for a prompt of an LLM or for a request to a decision model. Previously, such a character was removed together with the control characters, and the words around it were joined: `Fee:<tab>5,000 EUR` was sent as `Fee:5,000 EUR`, and a paragraph or a sentence with a line break inside it, e.g. one segmented with a SaT model or passed as a `Paragraph`, was sent with the last word of a line joined to the first word of the next line. The same applies to the form feed, the Unicode line and paragraph separators, and the narrow no-break space. Line breaks are kept, as before, in the text of a document that is sent as a whole. A text is sent unchanged where no words were joined.
- A negative `max_paragraphs_to_analyze_per_call` or `max_images_to_analyze_per_call` passed to an extraction method now raises `ValueError` before any requests are sent. Previously, an LLM was sent no requests and extracted nothing, without an error, when such a value applied to what it was to extract (`extract_all()` then failed with an unrelated error on an aspect that is not yet processed).
- A concept with references to sentences (`add_references=True` with `reference_depth="sentences"`) is now extracted from an aspect also when the sentences of the document are segmented only after the aspects were extracted, e.g. when the concept is attached to the aspect after `extract_aspects_from_document()` was called. Previously, for such a concept of a sub-aspect, `extract_concepts_from_aspect()` failed with an error that the sub-aspect must be assigned to the document, and succeeded when called again. For such a concept of an aspect of a document restored from its serialized form (`from_dict()`, `from_json()`, `from_disk()`), the prompt had no text of the paragraphs of the aspect. The paragraphs of a sub-aspect, and of the aspects of a restored document, are copies of the paragraphs of the document, and they did not get the sentences.
- The `reference_sentences` of an aspect with `reference_depth="sentences"` are now the sentences extracted for the aspect, i.e. the reference sentences of its extracted items, in the order of the items, each once. Previously, they were all the sentences of each reference paragraph of the aspect, including the sentences that were not extracted: a paragraph of five sentences of which one was extracted gave five reference sentences. The `reference_sentences` of the extracted items and the `reference_paragraphs` of the aspect are not affected, and the sub-aspects and the concepts of the aspect are extracted from its reference paragraphs, as before.

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
