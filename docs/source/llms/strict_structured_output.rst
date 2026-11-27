.. meta::
   :description: Use LiteLLM to request strict JSON Schema responses when extracting concepts or aspects with an LLM.
   :og:title: Strict JSON Schema responses

Strict JSON Schema Responses
============================

By default, ContextGem asks the LLM to follow its extraction prompt and then validates
the returned JSON itself. To also have a compatible provider enforce the response
shape, set ``strict_structured_output=True`` on :class:`~contextgem.public.llms.DocumentLLM`:

.. code-block:: python

   llm = DocumentLLM(
       model="openai/gpt-4o-mini",
       strict_structured_output=True,
   )
   llm.extract_concepts_from_document(document)

ContextGem sends LiteLLM an OpenAI-compatible ``response_format`` using a strict JSON
Schema for LLM concept and aspect extraction. The response is wrapped in an object with
an ``items`` array because strict JSON Schema responses require an object at the root.
ContextGem removes that wrapper and runs its usual validation, including concept-specific
checks such as parsing dates, checking rating ranges and validating references against the
document.

This setting is opt-in and does not apply to ``chat()``. Keep it disabled for providers
that do not support strict JSON Schema responses. If the provider rejects the schema,
ContextGem reports the request failure; it does not retry after dropping
``response_format``. Every LLM in a fallback chain must use the same setting.

Some user-defined ``JsonObjectConcept`` shapes cannot be expressed by strict JSON Schema:
objects with arbitrary keys or optional fields are rejected before the request is sent.
Use fixed, required fields for those concepts, or leave strict mode disabled.

Strict output constrains the response format. It does not establish that the extracted
values are true or supported by the document; ContextGem's existing extraction and
reference checks still apply.

LiteLLM's ``response_format`` parameter is documented in its
`JSON mode guide <https://docs.litellm.ai/docs/completion/json_mode>`_.
