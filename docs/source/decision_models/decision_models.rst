..
   ContextGem

   Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

:og:description: ContextGem: Decision Models


Decision Models
================

A decision model takes a text and typed questions about it (yes/no, pick one option, rate on a scale) and returns a probability for each possible answer. It generates no text.

ContextGem uses decision models next to LLMs. A :class:`~contextgem.public.decision_models.DocumentDecisionModel` extracts:

- :class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.LabelConcept` and :class:`~contextgem.public.concepts.RatingConcept`, from a document or from an aspect;
- aspects, by deciding for each paragraph whether it belongs to the aspect and, for an aspect with ``reference_depth="sentences"``, for each sentence of such a paragraph.

It has the same extraction methods as :class:`~contextgem.public.llms.DocumentLLM` (see :doc:`../llms/llm_extraction_methods`), and no ``chat()`` method. It can be used on its own, in a :class:`~contextgem.public.models.DocumentModelGroup` together with LLMs, and in a fallback chain with LLMs. Each item it extracts has the probabilities of all the possible answers in its ``probabilities`` attribute.

The requests are sent with `LiteLLM <https://github.com/BerriAI/litellm>`_, as for LLMs.


.. _decision-models-when-to-use-label:

🧭 When to Use a Decision Model
---------------------------------

.. list-table:: A decision model compared with an LLM
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Decision model
     - LLM
   * - Returns
     - A probability for each possible answer to a question
     - Generated text, which ContextGem parses into items
   * - Concepts
     - Boolean, label and rating concepts
     - All concept types
   * - Aspects
     - Yes, one paragraph at a time, with paragraphs or sentences as the extracted items
     - Yes, with paragraphs or sentences as the extracted items
   * - Justifications
     - No
     - Yes
   * - References for concepts
     - No
     - Yes
   * - Extraction from images
     - No
     - Yes, with a model that supports vision
   * - Probabilities of the answers
     - Yes
     - No

**A decision model is a good fit when:**

- **The answer comes from a closed set**: yes or no, one or several of your labels, a level of a scale.
- **You need to know how sure the model is of an answer**, e.g. to send the uncertain answers to a human review, or to accept a "yes" only above a threshold of your choice.
- **There are many such questions, or many documents, and the cost of generated output matters.** A decision model generates no text. As of October 2026, TypeSafe lists $0.042 per 1M input tokens for ``jev-1.13.0``, with output tokens free (`TypeSafe models <https://docs.typesafe.ai/models>`_), and OpenAI lists $0.10 per 1M input tokens for ``gpt-6-luna`` on the Decisions API, with input tokens as the only charge (`OpenAI Decisions guide <https://developers.openai.com/api/docs/guides/decisions>`_). What an extraction costs also depends on how many requests the text is sent in, see :ref:`decision-models-cost-label`.

**Use an LLM instead when:**

- **The value is open.** A :class:`~contextgem.public.concepts.StringConcept`, :class:`~contextgem.public.concepts.NumericalConcept`, :class:`~contextgem.public.concepts.DateConcept` or :class:`~contextgem.public.concepts.JsonObjectConcept` cannot be extracted by a decision model.
- **You need justifications, or references for concepts.** A decision model does not explain an answer and does not point to the parts of the text behind it.
- **The concept is extracted from images.**
- **The answer takes several steps of reasoning.** For example, *"The liability cap is lower than the total fees"* requires finding two amounts and comparing them.
- **The answer depends on dates or arithmetic.** For example, *"The agreement runs for more than 18 months"*, when the text gives a start date and an end date.
- **The text is long and mostly unrelated to the question.**

The last three points are among the limitations that TypeSafe documents for its Jev 1.13 model (`Jev 1.13 jaggedness <https://docs.typesafe.ai/model-jaggedness/jev-1.13>`_): accuracy falls as the text grows with content unrelated to the decision, dates are read as text and not as ordered quantities, counting and arithmetic are unreliable, and a question that requires several hops of reasoning costs accuracy. Other decision models may behave differently. To give the model less unrelated text, attach a concept to an aspect: it is then decided on the paragraphs of the aspect, not on the whole document.

.. note::
   Before you replace an LLM with a decision model, compare the two on a sample of your own documents for which you know the answers. You do not have to choose one for everything: in a model group, a decision model can take the closed questions, while an LLM extracts the rest (see :ref:`decision-models-model-group-label`).


🚀 Quick Start
----------------

The example below extracts four concepts from a document with a decision model, in one request, and reads the probabilities of the extracted items:

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_concepts.py
   :language: python
   :caption: Extracting concepts with a decision model


.. _decision-models-providers-label:

🔌 Supported Providers and Models
-----------------------------------

A decision model is identified in the same way as an LLM, in the format ``<provider>/<model_name>``. ContextGem supports the providers that LiteLLM serves through its Decisions API. With the pinned ``litellm`` version (1.104.2), these are:

.. list-table:: Providers of decision models
   :header-rows: 1
   :widths: 16 34 50

   * - Provider
     - ``model``
     - What it needs
   * - TypeSafe
     - ``typesafe/jev-latest``, ``typesafe/jev-1.13.0``, ``typesafe/jev-preview``
     - ``api_key``.
   * - OpenAI
     - ``openai/gpt-6-luna``
     - ``api_key``. The requests go to the Decisions API of OpenAI, which was in public beta as of October 2026. The same model name is used for chat and extraction with :class:`~contextgem.public.llms.DocumentLLM`: it is the class, not the model name, that makes it a decision model.
   * - Perplexity
     - ``perplexity/pplx-decider-v1-27b``
     - ``api_key``.
   * - OpenRouter
     - ``openrouter/typesafe/jev-1.13``
     - ``api_key``.
   * - Cloudflare
     - ``cloudflare/clef``, ``cloudflare/clef-flash``
     - ``api_key``, and the ID of your account: either in ``api_base``, as ``https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/run``, or in the ``CLOUDFLARE_ACCOUNT_ID`` environment variable.
   * - Local or self-hosted server
     - ``strands_decider/<model_name>``
     - ``api_base``: the address of the server. No API key is needed. The requests are sent to ``POST <api_base>/v1/systemone``. This is how a decision model served by Ollama or LM Studio is reached, see :ref:`decision-models-local-label`.

The model names above are the ones in the model list of this ``litellm`` version.

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_init.py
   :language: python
   :caption: Creating decision models

What to know about the providers:

- **The provider is checked when a decision model is created.** A model of a provider that the installed LiteLLM version serves no decision models of, e.g. ``anthropic/...``, is rejected with a validation error (a ``ValueError``) that lists the supported providers, also when ``model`` is assigned later. The model name is not checked: a name that the provider does not serve, a missing API key and a missing API base all fail at the first request, with ``ModelAPIError`` (see :ref:`decision-models-errors-label`).
- **API keys in the environment.** If ``api_key`` is not passed, LiteLLM reads the key from the environment variable of the provider, if it is set, e.g. ``TYPESAFE_API_KEY`` or ``OPENAI_API_KEY``.
- **What is tested.** The tests of ContextGem send requests to TypeSafe (``typesafe/jev-latest``), to the Decisions API of OpenAI (``openai/gpt-6-luna``), and to the local servers of Ollama (``strands_decider/nimble``) and LM Studio (``strands_decider/kev-4b``). The other providers are reached through the same LiteLLM call and are not tested with ContextGem.

ContextGem sends only text to a decision model. Concepts with the ``"image"`` or ``"multimodal"`` modality are not supported, whichever provider is used.


.. _decision-models-local-label:

💻 Local Decision Models
--------------------------

A decision model that runs on your machine is reached through the server that serves it. Both `Ollama <https://ollama.com/>`_ (since its version 0.35) and `LM Studio <https://lmstudio.ai/>`_ (since its version 0.4.26) serve decision models at ``POST /v1/systemone``. Create the decision model with the ``strands_decider/`` prefix and with the address of the server as ``api_base``:

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_local.py
   :language: python
   :caption: Using a local decision model

What to know about local decision models:

- **The prefix is not the one of local LLMs.** LiteLLM has no provider of decision models under the ``ollama/``, ``ollama_chat/`` and ``lm_studio/`` prefixes, which are for LLMs. A decision model created with such a prefix is rejected with a validation error (a ``ValueError``) that names the prefix to use. ``strands_decider/`` is the prefix that LiteLLM has for a self-hosted server with the ``/v1/systemone`` endpoint, whichever server it is.
- **No API key and no prices.** A local server needs no ``api_key``. The tokens are counted as for any other decision model, and no cost is calculated unless you set ``pricing_details``.
- **LM Studio serves only the models that it lists as decision models**, e.g. the GGUF builds that ``ggml-org`` publishes on Hugging Face, such as Kev 4B (``lms get https://huggingface.co/ggml-org/Kev-4B-GGUF``). A request for a model that LM Studio lists as an LLM is rejected, and the extraction fails with ``ModelAPIError``. The local server of LM Studio Bionic, which is a separate app, does not serve the endpoint as of its version 1.1.7 (October 2026).
- **The choice of the model matters.** In the tests of ContextGem on a short and unambiguous document, ``nimble`` (9B, with Ollama 0.40) answered each question as expected, with probabilities above 0.99. ``kev-4b`` (4B, with LM Studio 0.4.26) answered as expected except for one of the two labels of a multi-label concept that apply, which got a probability of 0.31. The much smaller ``tev1:0.8b`` answered several of the same questions wrongly. Check a local model on your own documents before you rely on it.
- **Shorter texts than with the cloud models.** A local model has a smaller context, and the text is not cut to fit it: both servers reject a request whose text and question exceed the context that the model is loaded with, Ollama also a request whose body exceeds 64 KiB, and the extraction fails with ``ModelAPIError``. By default, a document-level concept is decided on the text of the whole document, so send the text of a long document in parts, with ``max_paragraphs_to_analyze_per_call``, or attach its concepts to aspects (see :ref:`decision-models-limits-label`).


📝 Configuration Parameters
-----------------------------

The :class:`~contextgem.public.decision_models.DocumentDecisionModel` class accepts the following parameters, which have the same meaning as for :class:`~contextgem.public.llms.DocumentLLM`:

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Parameter
     - Type
     - Default Value
     - Description
   * - ``model``
     - ``str``
     - (Required)
     - Model identifier in format ``<provider>/<model_name>``. See :ref:`decision-models-providers-label`.
   * - ``api_key``
     - ``str | None``
     - ``None``
     - API key for authentication. Not needed for a self-hosted server. Not included in the serialized data.
   * - ``api_base``
     - ``str | None``
     - ``None``
     - Base URL of the API endpoint. Required for a self-hosted server. Not included in the serialized data.
   * - ``role``
     - ``str | None``
     - ``None``
     - Free-form role label of the model (up to 64 printable characters), used for routing in a model group: the model extracts the aspects and concepts with the same ``model_role``. Ignored when the model is used on its own. For more details, see :ref:`model-roles-label`.
   * - ``timeout``
     - ``int``
     - ``600``
     - Timeout in seconds for a request. A request that times out is retried as a transient error.
   * - ``num_retries_failed_request``
     - ``int``
     - ``3``
     - Number of retries when a request fails with a transient error (request timeout, rate limit, server or connection error). Retries are performed with exponential backoff and jitter. Non-transient errors (e.g. invalid request, failed authentication) are not retried. See :ref:`llm-retries-label`.
   * - ``pricing_details``
     - ``ModelPricing | None``
     - ``None``
     - :class:`~contextgem.public.data_models.ModelPricing` object with the prices for cost calculation. Set ``output_per_1m_tokens=0.0`` for a model that does not charge for output. See :ref:`decision-models-cost-label`.
   * - ``auto_pricing``
     - ``bool``
     - ``False``
     - Enable automatic cost calculation using ``genai-prices``. Mutually exclusive with ``pricing_details``. Rejected for a decision model for which ``genai-prices`` has the prices of a text-generating model, i.e. a price of output tokens (e.g. ``gpt-6-luna``), and for self-hosted (``strands_decider/``) models. See :ref:`decision-models-cost-label`.
   * - ``auto_pricing_refresh``
     - ``bool``
     - ``False``
     - When ``auto_pricing`` is enabled, allow ``genai-prices`` to auto-refresh its cached pricing data at runtime.
   * - ``is_fallback``
     - ``bool``
     - ``False``
     - Indicates whether the model is a fallback model, i.e. is assigned to another model and is used when that model fails.
   * - ``fallback_models``
     - ``list[DocumentDecisionModel | DocumentLLM]``
     - ``[]``
     - Models to use as fallbacks if the current one fails, tried in the given order until one of them succeeds. A fallback model can be a decision model or an LLM. Each must have ``is_fallback=True`` and the same role as the current model. See :ref:`decision-models-fallbacks-label`.
   * - ``decision_threshold``
     - ``float``
     - ``0.5``
     - The lowest probability of "yes" (above 0, up to 1) with which a yes/no question is taken as answered "yes": a paragraph or a sentence belongs to an aspect, a Boolean concept is ``True``, a label of a multi-label concept is selected. Applies unless the aspect or the concept has a ``decision_threshold`` of its own. See :ref:`decision-models-threshold-label`.
   * - ``async_limiter``
     - ``AsyncLimiter``
     - ``AsyncLimiter(10, 10)``
     - Relevant when concurrency is enabled for extraction tasks. Controls the frequency of the requests sent concurrently. Defaults to allowing 10 acquisitions per 10-second period. The rate applies to all the requests to the decision model that are sent concurrently, i.e. the extraction calls that run at the same time share it. This matters most for aspect extraction, which sends one request per paragraph (see :ref:`decision-models-aspects-label`). See `aiolimiter documentation <https://github.com/mjpieters/aiolimiter>`_ for AsyncLimiter configuration details.

Apart from ``decision_threshold``, which only a decision model has, a decision model has none of the other parameters of :class:`~contextgem.public.llms.DocumentLLM`, e.g. those that control the generation of text or the chat (``system_message``, ``output_language``, ``temperature``, ``top_p``, ``seed``, ``max_tokens``, ``reasoning_effort``, ``tools``), or ``max_retries_invalid_data``. Passing one of them raises a validation error.


.. _decision-models-supported-label:

✅ Supported Aspects and Concepts
-----------------------------------

A decision model can extract only what can be answered with a probability. An aspect or a concept that it cannot extract by its type or by its settings raises ``ValueError`` when an extraction method is called, before any request is sent.

.. list-table:: What a decision model can extract
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect or concept
     - Supported
     - Raises ``ValueError``
   * - :class:`~contextgem.public.aspects.Aspect`, including sub-aspects
     - With ``reference_depth="paragraphs"`` (default) or ``reference_depth="sentences"``, without justifications
     - ``add_justifications=True``
   * - :class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.LabelConcept` (both classification types), :class:`~contextgem.public.concepts.RatingConcept`
     - With ``modality="text"`` (default), without justifications and without references. The concept can be attached to a document or to an aspect.
     - ``modality="image"`` or ``modality="multimodal"``; ``add_justifications=True``; ``add_references=True``; more labels of a single-label concept, or more levels of a rating scale, than ``litellm`` accepts in a question; ``singular_occurrence=True`` when ``max_paragraphs_to_analyze_per_call`` is passed to the extraction method
   * - :class:`~contextgem.public.concepts.StringConcept`, :class:`~contextgem.public.concepts.NumericalConcept`, :class:`~contextgem.public.concepts.DateConcept`, :class:`~contextgem.public.concepts.JsonObjectConcept`
     - Not supported
     - Always

The number of the labels of a single-label concept and the length of a rating scale are limited by ``litellm``. A concept over such a limit is rejected with the same ``ValueError``, before any request is sent (see :ref:`decision-models-limits-label`).

A concept with ``singular_occurrence=True`` is supported unless ``max_paragraphs_to_analyze_per_call`` (above 0) is passed to the extraction method. With that parameter, a text is sent in parts, and a decision model returns an item for each part, so the concept would get several items. It is rejected with the same ``ValueError``, before any request is sent, whatever the number of parts the text would be split into (see :ref:`sending a long text in parts <decision-models-parts-label>`).

The error names the model and each aspect or concept that cannot be extracted, with the reasons:

.. code-block:: text
   :caption: The error for a concept that a decision model cannot extract

   ValueError: Decision model `typesafe/jev-latest` cannot extract: concept `Fee` (type `StringConcept` is not supported). A decision model extracts aspects, and Boolean, label and rating concepts from text without references, both without justifications. Route the other aspects and concepts to an LLM, e.g. in a model group.

**Which models are checked:**

- **A decision model used on its own** is given all the aspects and concepts that the extraction method processes, regardless of their ``model_role``, as a single LLM is. Each of them has to be supported.
- **A decision model in a model group** is given the aspects and concepts routed to it by the role (see :ref:`decision-models-model-group-label`). Only these are checked.
- **A decision model that is a fallback model** has to support everything that is given or routed to the model it is a fallback for (see :ref:`decision-models-fallbacks-label`).

**What each extraction method checks:** what it is to extract.

- ``extract_all()`` checks all the aspects and concepts of the document, including the sub-aspects and the concepts of the aspects, before the first request. It therefore works with a decision model alone only if everything attached to the document is supported. To extract a document that also has aspects or concepts that need an LLM, use a model group.
- ``extract_aspects_from_document()`` checks the aspects and their sub-aspects, and not the concepts of the aspects.
- ``extract_concepts_from_document()`` checks the concepts of the document, and ``extract_concepts_from_aspect()`` the concepts of the aspect and of its sub-aspects.


.. _decision-models-questions-label:

❓ How Concepts Become Questions
----------------------------------

Each concept is turned into one or more typed questions. The text of a question is built from the ``name`` and the ``description`` of the concept, and the answer is converted to the value of the extracted item:

.. list-table:: Questions and item values by concept type
   :header-rows: 1
   :widths: 25 35 40

   * - Concept
     - Question sent to the model
     - Value of the extracted item
   * - :class:`~contextgem.public.concepts.BooleanConcept`
     - One yes/no question: is the statement in the description true according to the text?
     - ``True`` if the probability of "yes" is the decision threshold (0.5 by default) or more, else ``False``
   * - :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_class"``
     - One question with the labels as the options, of which one is to be picked
     - A list with the most probable label, e.g. ``["Written"]``
   * - :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_label"``
     - One yes/no question per label: does the label apply to the text?
     - A list of the labels with a probability of the decision threshold (0.5 by default) or more, in the order of the labels of the concept. No item is extracted if no label reaches the threshold.
   * - :class:`~contextgem.public.concepts.RatingConcept`
     - One rating question, with one level for each integer of the rating scale
     - The most probable level, as an integer of the scale

What follows from this:

- **A Boolean concept always gets an answer.** The item is ``True`` or ``False``: the model does not abstain. "No" covers both a text that contradicts the statement and a text that says nothing about it. An LLM, in contrast, may return no item for a Boolean concept that cannot be determined from the text.
- **Word the description of a Boolean concept as a statement** that is either true or false, e.g. *"Either party may terminate the agreement without cause"*.
- **A single-label concept always gets a label**, even if none fits well. Add a label such as ``"Other"`` or ``"Not specified"`` if that is a possible outcome.
- **A rating is the most probable level**, not an average over the levels. The levels are sent to the model as numbers (*"3 on a scale from 1 to 5"*), so describe in the concept description what the low and the high end of the scale mean.
- **Labels are sent as they are.** A label has no description of its own, so make the labels self-explanatory.
- **If two answers are equally probable**, the first one is taken: the first label in the order of the labels of the concept, or the lower level of the rating scale.
- **At most one item is extracted** for a concept from a document or an aspect, with ``singular_occurrence=True`` and without it, unless the text is sent in parts: a concept then has one item for each part, and ``singular_occurrence=True`` is not accepted (see :ref:`below <decision-models-parts-label>`).

By default, all the questions of the concepts processed in one call are sent in one request, together with the text: the text of the document for document-level concepts, or the text of the paragraphs extracted for the aspect for aspect-level concepts. As for LLMs, ``max_items_per_call`` sets how many concepts are processed in one call (all of them by default, or one with ``use_concurrency=True``). With a local model, the number of the questions in a request can change the answers (see :ref:`decision-models-items-per-request-label`).

.. _decision-models-parts-label:

**A long text can be sent in parts.** With ``max_paragraphs_to_analyze_per_call`` passed to the extraction method, a text with more paragraphs than that is not sent as a whole. Its paragraphs (those of the document for document-level concepts, those extracted for the aspect for aspect-level concepts) are split into consecutive parts of at most that many paragraphs, in the same way as for an LLM, and one request is sent for each part, with the text of the part and the questions of all the concepts of the call. A text with no more paragraphs than that is sent as a whole, as without the parameter.

- **A concept then has one item for each part**, in the order of the parts, each with the probabilities of the answers on its part. The answers on the parts are not merged into one item: combining the items is up to you, e.g. taking a Boolean concept as true if it is true for any part, or taking the highest rating.
- **Each part is decided on its own text.** The model is not sent the other parts. A Boolean concept is ``False`` for a part that does not support the statement, and a single-label and a rating concept get a label and a level for each part, including the parts that say nothing about the concept. A multi-label concept has no item for a part in which no label reaches the threshold, so it can have fewer items than there are parts.
- **An item does not tell which part it comes from.** As any item of a concept extracted by a decision model, it has no references.
- **A concept with** ``singular_occurrence=True`` **is rejected.** An LLM stops at the first part that gives an item for such a concept, while a decision model answers on every part. When ``max_paragraphs_to_analyze_per_call`` is above 0, ``ValueError`` is therefore raised, before any request is sent, for each such concept that a decision model is to extract, on its own or as a fallback model (see :ref:`decision-models-supported-label`). This does not depend on the length of the text: the concept is also rejected if the text has few enough paragraphs to be sent as a whole. To extract the concept, set ``singular_occurrence=False`` and combine the items yourself, leave ``max_paragraphs_to_analyze_per_call`` out, or route the concept to an LLM.
- **The requests for the parts** are sent one after another, or concurrently with ``use_concurrency=True``, at the rate allowed by the ``async_limiter`` of the decision model, as the requests for the paragraphs in aspect extraction (see :ref:`decision-models-aspects-label`).

.. code-block:: python
   :caption: Extracting a concept from a long text in parts

   # Each request has the text of up to 50 paragraphs
   decision_model.extract_concepts_from_document(
       document, max_paragraphs_to_analyze_per_call=50
   )

   # One item for each part: combine them as your task requires
   is_true_for_any_part = any(item.value for item in boolean_concept.extracted_items)
   highest_p_yes = max(
       item.probabilities[True] for item in boolean_concept.extracted_items
   )


.. _decision-models-probabilities-label:

🎲 Probabilities of Extracted Items
-------------------------------------

Each item extracted by a decision model has the ``probabilities`` attribute: a dictionary that maps every possible answer to its probability, a number from 0 to 1. The keys are the answers as they appear as item values.

.. list-table:: The ``probabilities`` attribute by item
   :header-rows: 1
   :widths: 25 40 35

   * - Item of
     - ``probabilities``
     - Example
   * - :class:`~contextgem.public.concepts.BooleanConcept`
     - The probability ``p`` of "yes" under ``True``, and ``1 - p`` under ``False``
     - ``{True: 0.83, False: 0.17}``
   * - :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_class"``
     - The probability of each label of the concept
     - ``{"Written": 0.71, "Email": 0.2, "Any form": 0.06, "Not specified": 0.03}``
   * - :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_label"``
     - For each label of the concept, the probability of its own yes/no question. Each label is decided separately, so the probabilities do not sum to 1.
     - ``{"NDA": 0.97, "Confidential": 0.62, "Employment": 0.08}``, for the item value ``["NDA", "Confidential"]``
   * - :class:`~contextgem.public.concepts.RatingConcept`
     - The probability of each integer of the rating scale
     - ``{1: 0.0, 2: 0.0, 3: 0.15, 4: 0.55, 5: 0.3}``
   * - :class:`~contextgem.public.aspects.Aspect`
     - The probability ``p`` that the paragraph, or the sentence for an aspect with ``reference_depth="sentences"``, belongs to the aspect under ``True``, and ``1 - p`` under ``False``
     - ``{True: 0.91, False: 0.09}``

The dictionary has all the possible answers, in the order of the labels of the concept or of the levels of the scale, and the keys keep their types: ``True`` and ``False`` for a yes/no question, strings for labels, integers for ratings. The probabilities of the labels of a single-label concept and of the levels of a rating are stored as the model returns them, without being normalised.

.. code-block:: python
   :caption: Reading the probabilities

   # Boolean concept: the probability of "yes"
   item = boolean_concept.extracted_items[0]
   p_yes = item.probabilities[True]

   # Rating concept: the probability of the returned rating
   item = rating_concept.extracted_items[0]
   p_rating = item.probabilities[item.value]

   # Label concept: the probability of each returned label
   item = label_concept.extracted_items[0]
   p_labels = {label: item.probabilities[label] for label in item.value}

What is not extracted has no probabilities to read. A multi-label concept none of whose labels reaches the decision threshold has no extracted item, and a paragraph with a probability below the threshold is not among the extracted items of the aspect. To have them extracted, lower the threshold (see :ref:`decision-models-threshold-label`). The answers of the model to each request, including these, are recorded as JSON in the usage of the model (see :ref:`decision-models-cost-label`).

``probabilities`` is ``None`` for every item extracted by an LLM, including an LLM that is a fallback of a decision model. The probabilities are kept when a document is serialized (see :doc:`../serialization`).


.. _decision-models-threshold-label:

🎚️ Decision Threshold
-----------------------

A yes/no question is taken as answered "yes" if the probability of "yes" is the decision threshold or more. The threshold is 0.5 by default, i.e. "yes" is the more probable answer. It decides three things:

- which paragraphs, or which sentences with ``reference_depth="sentences"``, become the extracted items of an :class:`~contextgem.public.aspects.Aspect`;
- whether the value of a :class:`~contextgem.public.concepts.BooleanConcept` is ``True`` or ``False``;
- which labels a :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_label"`` gets.

A lower threshold extracts more, including what the model is less sure of. A higher threshold extracts only what the model is surer of. Set it in two places:

- **For the decision model**, with its ``decision_threshold`` parameter: it applies to all the aspects and concepts that the model extracts.
- **For an aspect or a concept**, with its own ``decision_threshold`` parameter: it applies instead of the threshold of the decision model, whichever decision model extracts the aspect or the concept. It is ``None`` by default, i.e. the threshold of the decision model applies.

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_threshold.py
   :language: python
   :caption: Setting the decision threshold

What to know about the threshold:

- **It does not apply to single-label and rating concepts.** For a :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_class"`` and for a :class:`~contextgem.public.concepts.RatingConcept`, the most probable option is taken, whatever its probability is. Setting ``decision_threshold`` for a single-label concept raises a validation error (a ``ValueError``), and a rating concept has no such parameter.
- **It does not change the probabilities.** The ``probabilities`` of an item are those returned by the model, whatever threshold its value is set with: e.g. with a threshold of 0.7 and a probability of 0.6, the value of a Boolean concept is ``False`` and its probabilities are ``{True: 0.6, False: 0.4}``.
- **A sub-aspect does not get the threshold of its aspect.** Each aspect has its own threshold, or that of the decision model.
- **An LLM does not use it.** The ``decision_threshold`` of an aspect or a concept is ignored when an LLM extracts it, including an LLM that is a fallback of a decision model. A fallback decision model applies its own ``decision_threshold`` to the aspects and concepts that have none of their own.
- **The value must be above 0 and not above 1.** With a threshold of 1, only a probability of exactly 1 counts as "yes".
- **The probabilities are what the model reports**, so the right threshold depends on the model: check it on your own documents (see :ref:`decision-models-trust-label`).

See :ref:`decision-models-trust-label` for what a probability does and does not tell.


.. _decision-models-aspects-label:

📄 Extracting Aspects
-----------------------

A decision model extracts aspects one paragraph at a time. The paragraphs are the extracted items of an aspect with ``reference_depth="paragraphs"`` (default), for which the following applies. For an aspect with ``reference_depth="sentences"``, the sentences of these paragraphs are then decided on as well (see "Sentences as the extracted items" below):

- **One request per paragraph.** A request has the text of one paragraph and, for each aspect processed in the call, one yes/no question: is the paragraph relevant to the aspect? The question is built from the ``name`` and the ``description`` of the aspect.
- **A paragraph is judged on its own.** The model is not sent the neighbouring paragraphs or the rest of the document, whereas an LLM is sent the paragraphs together. A paragraph that can be understood only in its context, e.g. *"The same applies to the Supplier."*, can therefore be judged differently than by an LLM.
- **The additional context of a paragraph is sent with it.** If a paragraph has ``additional_context``, e.g. its heading level, list context or table position (see :ref:`paragraph-additional-context`), the text of its request has both, each with a label, and the question tells the model that the text has the additional context of the paragraph, and that the paragraph is what it is to judge. A paragraph without additional context is sent as its text alone. As a decision model is not sent the neighbouring paragraphs, the additional context is the only context of a paragraph that it gets.
- **A paragraph with a probability of the decision threshold or more becomes an extracted item** of the aspect. The threshold is 0.5 by default, and can be set for the decision model and for each aspect (see :ref:`decision-models-threshold-label`). As for LLMs, the value of the item is the text of the paragraph, and ``reference_paragraphs`` of the item has the paragraph. A paragraph can belong to several aspects.
- **An aspect that no paragraph belongs to has no extracted items.** Its sub-aspects and concepts are then not extracted, as for LLMs.

The text of the request for a paragraph that has additional context looks like this:

.. code-block:: text
   :caption: The text sent for a paragraph with additional context

   Paragraph: The same applies to the Supplier.

   Additional context of the paragraph: Section 7. Confidentiality; list item 2 of 3

The sub-aspects of an aspect are extracted in the same way from the paragraphs extracted for that aspect, with one request for each of these paragraphs. The concepts of an aspect are decided on the text of the paragraphs extracted for it.

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_aspects.py
   :language: python
   :caption: Extracting aspects and their concepts with a decision model

**Sentences as the extracted items.** An aspect with ``reference_depth="sentences"`` is extracted in two stages:

1. **Paragraphs.** One request per paragraph, as described above: does the paragraph belong to the aspect? The text and the question are the same as for an aspect with ``reference_depth="paragraphs"``.
2. **Sentences.** For each paragraph that belongs to the aspect, one request per sentence of the paragraph: is the sentence relevant to the aspect? The text of the request has the sentence, which is what the model judges, and the paragraph that the sentence is from, as its context, each with a label. The ``additional_context`` of the sentence and that of the paragraph follow, if they are set. A request has one question for each aspect with ``reference_depth="sentences"`` of the call that the paragraph belongs to.

.. code-block:: text
   :caption: The text sent for a sentence

   Sentence: The notice shall be given in writing.

   Paragraph that the sentence is from: Either party may terminate this agreement with a notice of 30 days. The notice shall be given in writing.

   Additional context of the paragraph: Section 3. Termination

What to know about the aspects with sentences as the extracted items:

- **A sentence with a probability of the decision threshold or more becomes an extracted item** of the aspect. The same threshold applies at both stages: to the paragraphs, and to their sentences. As for LLMs, the value of the item is the text of the sentence, ``reference_sentences`` of the item has the sentence, and ``reference_paragraphs`` has its paragraph. The ``probabilities`` of the item are those of the sentence.
- **A paragraph that belongs to the aspect gives no item itself.** If none of its sentences belongs to the aspect, the paragraph is not among the paragraphs of the aspect, i.e. the sub-aspects and the concepts of the aspect are not extracted from it.
- **The number of requests** is the number of the paragraphs plus the number of the sentences in the paragraphs that belong to the aspect. For example, a document of 100 paragraphs, of which 10 belong to the aspect, with about 3 sentences each, takes about 130 requests: 100 for the paragraphs, and about 30 for the sentences. The text of a paragraph that belongs to the aspect is sent, and billed, once more with each of its sentences.
- **The sentences are segmented before the extraction**, as for LLMs. A sentence segmented from a paragraph has the ``additional_context`` of the paragraph, which is sent once with the sentence, as that of the paragraph.
- **The requests for the sentences are sent after those for the paragraphs**, one after another or, with ``use_concurrency=True``, concurrently, as the requests for the paragraphs are.
- **Aspects with different reference depths are processed in separate calls**, as for LLMs: each call sends its own request for each paragraph.

**Use concurrency.** A document of 300 paragraphs takes 300 requests. By default, they are sent one after another. With ``use_concurrency=True`` passed to the extraction method, they are sent concurrently, at the rate allowed by the ``async_limiter`` of the decision model. The default limiter allows 10 requests per 10 seconds, at which 300 requests take about five minutes, so set it to the rate limits of your provider account:

.. code-block:: python
   :caption: Setting the rate of concurrent requests

   from aiolimiter import AsyncLimiter

   from contextgem import DocumentDecisionModel

   decision_model = DocumentDecisionModel(
       model="typesafe/jev-latest",
       api_key="<your-typesafe-api-key>",
       async_limiter=AsyncLimiter(50, 1),  # up to 50 requests per second
   )

   # document = decision_model.extract_all(
   #     document, use_concurrency=True, max_items_per_call=10
   # )

.. note::
   **When you enable concurrency, also set the max items per call.** As for LLMs, each aspect and each concept is processed in a call of its own when ``use_concurrency=True`` is passed and ``max_items_per_call`` is not set. With a decision model, each text is then sent, and billed, once per aspect or concept: 3 aspects and 100 paragraphs take 300 requests instead of 100. To send each text once, with the questions of all the aspects or concepts, set ``max_items_per_call`` to the number of the aspects or concepts, or more. With a local model, check first that the answers do not change with the number of the questions in a request (see :ref:`decision-models-items-per-request-label`).

``max_paragraphs_to_analyze_per_call`` changes nothing in aspect extraction, as each paragraph is sent in a request of its own in any case. For concepts, it sets how many paragraphs a request has at most, i.e. in how many parts a text is sent (see :ref:`sending a long text in parts <decision-models-parts-label>`).

The length of the document does not limit aspect extraction, as each request has one paragraph only.


.. _decision-models-model-group-label:

🤖🤖 Using a Decision Model in a Model Group
----------------------------------------------

In a :class:`~contextgem.public.models.DocumentModelGroup`, each aspect and concept is extracted by the model whose ``role`` equals its ``model_role`` (see :ref:`model-roles-label`). Give the decision model a role, and assign the same ``model_role`` to the aspects and concepts it is to extract. The LLM in the group extracts the rest, e.g. the concepts with open values:

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_group.py
   :language: python
   :caption: A decision model and an LLM in a model group

The routing is yours to declare: ContextGem does not pick a model for an aspect or a concept. If an aspect or a concept that a decision model cannot extract is routed to it, ``ValueError`` is raised before any request is sent (see :ref:`decision-models-supported-label`): set the ``model_role`` of such an aspect or concept to the role of an LLM in the group.

What else applies to a group with a decision model:

- **An aspect and what is attached to it can go to different models.** For example, an LLM extracts an aspect, and the decision model decides the Boolean concepts of that aspect on the paragraphs that the LLM extracted. The same holds for an aspect and its sub-aspects.
- **A decision model can be the model without a role** in a group, in which case it is given the aspects and concepts without ``model_role``. A group can also consist of decision models only.
- **The output language of a group applies to its LLMs.** The ``output_language`` of the group has to be the one of each LLM in it. A decision model has no output language.
- **Usage and cost** of the decision model are read as for any model of a group, e.g. ``model_group.get_usage(model_role="decide")``.


.. _decision-models-fallbacks-label:

🔄 Fallback Chains
--------------------

The ``fallback_models`` of a decision model and of an LLM can hold models of both kinds. They are tried in the given order when the primary model fails:

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_fallbacks.py
   :language: python
   :caption: Fallback chains with decision models and LLMs

**A decision model that falls back to an LLM** is always a valid chain, as an LLM can extract whatever a decision model can. The LLM extracts the aspects and concepts with its own prompts, so the items it extracts have no probabilities: check ``item.probabilities is None`` if your code relies on them. Unlike a decision model, an LLM may also return no item for a Boolean concept.

**An LLM that falls back to a decision model** is valid only if the decision model can extract everything that is given or routed to the LLM (see :ref:`decision-models-supported-label`). This is checked when an extraction method is called, before any request is sent, and not when the fallback model is assigned: if the LLM is to extract e.g. a :class:`~contextgem.public.concepts.StringConcept` or a concept with justifications, ``ValueError`` is raised. In a chat, a decision model among the fallback models of an LLM is skipped, and the fallback LLMs after it are still tried.

In both directions, each fallback model must be created with ``is_fallback=True``, have the same ``role`` as the primary model, and cannot have fallback models of its own. A fallback model must also differ from the primary model, and from the other fallback models, in at least one configuration parameter, unless it is a model of the other kind: the same model name, e.g. ``openai/gpt-6-luna``, can be used as a decision model and as its fallback LLM. The further requirements for the fallbacks of an LLM (the same output language, system message and tools, see :ref:`llm-fallback-llms-label`) apply between two LLMs only.

The next fallback model is tried when the current model fails for good: when its request fails with a non-transient error, when the retries of its request are exhausted, or when a decision model returns a response that cannot be used (see :ref:`decision-models-errors-label`). A decision model is not asked again for a response that cannot be used, whether it is the primary model or a fallback model, while a fallback LLM that returns invalid data is asked again as LLMs are. If there is no fallback model left to try, the error is the one of the last model that failed. Usage and cost are tracked separately for the primary model and for each of its fallback models.


.. _decision-models-cost-label:

📊 Usage and Cost
-------------------

Usage and cost are tracked as for LLMs, with ``get_usage()``, ``get_cost()`` and ``reset_usage_and_cost()``:

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_usage_and_cost.py
   :language: python
   :caption: Tracking the usage and cost of a decision model

Token usage is tracked for every decision model, while costs are calculated only if ``pricing_details`` or ``auto_pricing`` is set. Each request is recorded as a call: its ``prompt`` is the request, i.e. the model, the text and the questions, as JSON, and its ``response`` is the answers of the model as JSON. A request that failed is recorded as a call without a response.

**How pricing is set:**

- ``auto_pricing=True`` works for the models whose prices of decision requests ``genai-prices`` has, i.e. prices of input tokens without a price of output tokens. With the pinned ``genai-prices`` version (0.1.9), these are the TypeSafe models listed above.
- For a model that ``genai-prices`` has no price for, a warning is issued when the model is created with ``auto_pricing=True`` and when its usage is recorded, and the costs stay at zero. With ``genai-prices`` 0.1.9, this is the case for the decision models of Perplexity, OpenRouter and Cloudflare listed above: set ``pricing_details`` to get the costs of these models.
- ``auto_pricing=True`` is rejected with a validation error (a ``ValueError``) for a decision model for which ``genai-prices`` has the prices of a text-generating model, i.e. prices that include a price of output tokens, when the model is created and when ``auto_pricing`` or ``model`` is assigned. A decision model is billed for its input only, so such prices are those of the chat requests to a model of the same name, and the costs would be calculated with the wrong prices without any notice. With ``genai-prices`` 0.1.9, this is the case for ``gpt-6-luna``, whose chat prices differ from its prices on the Decisions API, whichever provider serves it (``openai/gpt-6-luna``, ``openrouter/openai/gpt-6-luna``). ``auto_pricing=True`` is also rejected for self-hosted models (``strands_decider/``), which have no prices. Set ``pricing_details`` for such a model instead.
- With ``pricing_details``, :class:`~contextgem.public.data_models.ModelPricing` requires both ``input_per_1m_tokens`` and ``output_per_1m_tokens``. Set ``output_per_1m_tokens=0.0`` for a provider that does not charge for output.

As of October 2026, TypeSafe lists $0.042 per 1M input tokens for ``jev-1.13.0`` (`TypeSafe models <https://docs.typesafe.ai/models>`_), and OpenAI lists $0.10 per 1M input tokens for ``gpt-6-luna`` on the Decisions API (`OpenAI Decisions guide <https://developers.openai.com/api/docs/guides/decisions>`_). Neither charges for output tokens. Check the pricing page of your provider for the current prices.

**What a request costs.** The text is sent, and billed as input tokens, once per request: the text of the document or of the paragraphs of the aspect for each call of concepts, and the text of the paragraph for each request of aspect extraction, including each request for a sentence of the paragraph, for an aspect with ``reference_depth="sentences"``. The fewer requests a text is sent in, the less it costs:

- With ``use_concurrency=True``, set ``max_items_per_call``, as each aspect and each concept is otherwise processed in a call of its own (see :ref:`decision-models-aspects-label`).
- The parameters that a decision model does not use do not split its calls: aspects and concepts that differ only in ``justification_depth`` or ``justification_max_sents``, and concepts that differ only in ``reference_depth``, are processed in the same call, while an LLM is called separately for each combination of these parameters.
- More than 128 questions on a text are sent in several requests (see :ref:`decision-models-limits-label`).
- With ``max_paragraphs_to_analyze_per_call``, the text of a call of concepts is spread over the requests for its parts, i.e. it is still sent once per call, while the questions of the call are sent, and billed, with each part (see :ref:`sending a long text in parts <decision-models-parts-label>`).

**A request is billed for more than its text.** The questions are billed as input tokens too, and the input tokens that a provider reports for a request can exceed those of its text and questions. For short texts, such as the paragraphs in aspect extraction, the number of the requests and of the questions can therefore weigh more in the cost than the length of the text. ``get_usage()`` shows the input tokens reported for each of your own requests.


.. _decision-models-items-per-request-label:

🔢 How Many Questions to Put in One Request
----------------------------------------------

All the questions of a call on a text are sent in one request, and ``max_items_per_call`` sets how many aspects or concepts a call has: all of them by default, or one with ``use_concurrency=True``. A Boolean, a single-label and a rating concept take one question each, a multi-label concept takes one question per label, and each aspect of the call takes one question in the request for a paragraph or for a sentence.

The more questions a request has, the fewer requests a text is sent in, and the less the extraction costs (see :ref:`decision-models-cost-label`). Whether the answer to a question depends on the other questions of the request is a property of the model:

- **Hosted Jev models are documented to answer each question on its own.** TypeSafe states that all the questions of a request see the same text and are evaluated independently and in parallel (`TypeSafe: building with System One <https://docs.typesafe.ai/concepts/how-to-build-with-system-one>`_), and a published experiment got the same probabilities for the same questions in requests with 6 to 400 questions (`jev-fanout <https://github.com/TheWebDevel/jev-fanout>`_). The tests of ContextGem do not check this. With such a model, process all the aspects, or all the concepts, in one call.
- **A local model can answer differently in a request with many questions.** In a test of ContextGem with ``nimble`` served by Ollama 0.40.2, the 7 questions of 4 concepts were sent on a short document in one request together with up to 50 other yes/no questions. When the 7 questions came after the other questions, the probability of "yes" for a concept that is clearly true fell with the number of the questions in the request: 0.99 with 7 questions, 0.96 with 37, 0.77 with 47, 0.38 with 52 and 0.05 with 57, i.e. the concept became false. When the 7 questions came first, their probabilities changed by less than 0.01. The input tokens grew from about 8,000 with 7 questions to about 400,000 with 57 questions, i.e. each question was processed together with all the others. In the same test, ``kev-4b`` served by LM Studio 0.4.26 gave the same values with 57 questions as with 7, in either order, with the probabilities within 0.03, and with the input tokens growing in proportion to the number of the questions.

These were single tests on one short document: they show that the effect exists, not how large it is with your model and your documents. With a local model, before you rely on requests with many questions, extract a few of your aspects or concepts one per call (``max_items_per_call=1``) and all in one call, and compare the probabilities. If they differ, lower ``max_items_per_call``.

.. code-block:: python
   :caption: Limiting the number of concepts per call

   # Each request has the questions of up to 20 concepts
   decision_model.extract_concepts_from_document(document, max_items_per_call=20)


.. _decision-models-limits-label:

📏 Limits
-----------

**Limits of litellm.** ``litellm`` accepts a request only within its limits. With the pinned ``litellm`` version (1.104.2), these are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Limit
     - What it means for the extraction
   * - Up to 128 questions per request
     - Not a limit for the extraction: ContextGem sends the questions over the cap in further requests on the same text. A Boolean, single-label or rating concept takes one question, and a multi-label concept takes one question per label, so e.g. a multi-label concept with 130 labels takes two requests. The text is sent, and billed, once per request. In aspect extraction, each aspect of the call takes one question on each paragraph.
   * - 2 to 255 options of a question with one option to pick
     - A :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_class"`` can have up to 255 labels: with more labels, ``ValueError`` is raised. A multi-label concept is not affected, as each of its labels is a question of its own.
   * - 2 to 10 levels of a rating question
     - The rating scale of a :class:`~contextgem.public.concepts.RatingConcept` can have up to 10 levels: ``rating_scale=(1, 10)`` works, while ``rating_scale=(0, 10)`` has 11 levels and raises ``ValueError``.

A concept over the limit on the options or on the levels is rejected as any other concept that a decision model cannot extract (see :ref:`decision-models-supported-label`): ``ValueError`` is raised when an extraction method is called, before any request is sent, and the other aspects and concepts of that extraction are not extracted either. The error names the concept and gives the reason of ``litellm``:

.. code-block:: text
   :caption: The error for a rating scale with more levels than litellm accepts

   ValueError: Decision model `typesafe/jev-latest` cannot extract: concept `Clarity` (its `score` question with 11 options is not accepted by litellm: `levels`: Value should have at most 10 items after validation, not 11). A decision model extracts aspects, and Boolean, label and rating concepts from text without references, both without justifications. Route the other aspects and concepts to an LLM, e.g. in a model group.

ContextGem has no limits of its own on the options and the levels: it checks each question with the request types of the installed ``litellm`` version, so the limits are those of that version. Use an LLM for a concept over a limit, e.g. by routing it to an LLM in a model group.

**The text of a request must fit the model.** By default, a document-level concept is decided on the text of the whole document, and an aspect-level concept on the text of all the paragraphs of the aspect, each sent in one request. The length of a text is not checked before the request is sent, and a text is not cut to fit. A request that the provider rejects for its length fails as any other rejected request. For Jev 1.13, TypeSafe documents a limit of 32k tokens for the text together with the longest question, and of 64k tokens per request (`TypeSafe models <https://docs.typesafe.ai/models>`_, as of October 2026). For a longer text, there are two ways, which can be combined:

- **Send the text in parts**, with ``max_paragraphs_to_analyze_per_call`` passed to the extraction method: each request then has the text of at most that many paragraphs, and each concept gets one item for each part (see :ref:`sending a long text in parts <decision-models-parts-label>`). A part is measured in paragraphs, not in tokens, so set the number by the length of your paragraphs.
- **Attach the concepts to aspects**: the aspects are extracted paragraph by paragraph, and the concepts of an aspect are decided on the paragraphs of the aspect only.

Providers may set further limits of their own. See the documentation of your provider, e.g. the `TypeSafe documentation <https://docs.typesafe.ai/>`_ or the `OpenAI Decisions guide <https://developers.openai.com/api/docs/guides/decisions>`_.


.. _decision-models-errors-label:

🚨 Errors
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Error
     - When it is raised
   * - ``ValueError``
     - When an extraction method is called, before any request is sent: the decision model, or a decision model in the fallback chain, cannot extract an aspect or a concept that it is given (see :ref:`decision-models-supported-label`), including a concept over a limit of ``litellm`` (see :ref:`decision-models-limits-label`). Also when a decision model is created, or its params are assigned, as a validation error: the provider of the model is not a provider of decision models that ``litellm`` supports (see :ref:`decision-models-providers-label`), or ``auto_pricing`` cannot be used for the model (see :ref:`decision-models-cost-label`).
   * - ``ModelAPIError``
     - A request failed for good: with a non-transient error, or with a transient error after ``num_retries_failed_request`` retries. This includes the requests that ``litellm`` rejects before sending them, e.g. for a missing API key or API base. The error has the number of retries made (``retry_count``) and the error of the request (``original_error``).
   * - ``ModelExtractionError``
     - A response cannot be used: the model refused to answer a question, an answer is missing, an answer does not have a valid probability for each of the options offered, or has a zero probability for each of them, or the option that an answer picks is not one of the options offered. The request is not sent again, as the same response is expected.

``ModelAPIError`` and ``ModelExtractionError`` are raised only if there is no fallback model left to try (see :ref:`decision-models-fallbacks-label`).

What to expect when a call fails:

- **One answer that cannot be used fails the whole call.** No aspect or concept of that call gets its items, including those whose answers were valid. The aspects and concepts extracted in other calls keep their items.
- **In aspect extraction, the answers on all the paragraphs are needed.** When a request for a paragraph fails for good, or its response cannot be used, the requests not yet sent are not sent, and no aspect of the call gets its items. The same holds for the requests for the sentences of the aspects with ``reference_depth="sentences"``, which are not sent at all if a request for a paragraph fails.
- **For concepts extracted from a text in parts, the answers on all the parts are needed.** When a request for a part fails for good, or its response cannot be used, the requests not yet sent are not sent, and no concept of the call gets its items, including for the parts already answered. The logged error names the part, e.g. ``on part 2 of 3 of the text``.
- **The requests made are recorded in the usage**, with their tokens, also when the call fails.
- **A request that fails for good stops the other calls of the model.** With ``use_concurrency=True``, the aspects or concepts are extracted in several calls at the same time, unless ``max_items_per_call`` puts them into one call. When ``ModelAPIError`` is to be raised for one of the calls, the requests in flight of the other calls are cancelled first, and no request is sent afterwards. The aspects and concepts extracted by then keep their items, and the requests completed by then are recorded in the usage.
- **A model that fails stops the other models of its group.** In a model group, with ``use_concurrency=True``, the models extract at the same time. When the extraction with one of them fails, the requests in flight of the other models are cancelled before the error is raised, and no request is sent afterwards. The aspects and concepts extracted by then keep their items, and the requests completed by then are recorded in the usage.
- **A warning instead of an exception.** With ``raise_exception_on_extraction_error=False`` passed to the extraction method, a warning is issued instead of ``ModelAPIError`` or ``ModelExtractionError``, and the aspects or concepts of the failed call are left without items. The ``ValueError`` above is raised in any case.


.. _decision-models-trust-label:

⚠️ How Far to Trust the Probabilities
---------------------------------------

- **A probability is the estimate of the model, not a guarantee.** A probability of 0.9 does not mean that the answer is correct in 9 cases out of 10 on your documents. Before you rely on a threshold, check on a sample of your documents how often the answers above it are correct.
- **Single-label answers can depend on the order of the labels.** TypeSafe documents for Jev 1.13 that the order of the options can affect the answer, and that the model leans toward the option that comes first (`Jev 1.13 jaggedness <https://docs.typesafe.ai/model-jaggedness/jev-1.13>`_). For ``gpt-6-luna`` on the OpenAI Decisions API, a user reports in the `announcement thread <https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877>`_ of the API that the order of the options changes the probabilities, and that the top option gets too much probability. ContextGem sends the labels in the order in which they are set on the concept, so the order of ``labels`` can affect the result.
- **Check a threshold for each model.** A threshold that you checked for one model has to be checked again for another model, including a fallback model.


💾 Serialization
------------------

A decision model is serialized and deserialized in the same way as an LLM, also as a part of a model group or of a fallback chain (see :doc:`../serialization`). Its ``api_key`` and ``api_base`` are never included in the serialized data, so set them again after deserialization (see :ref:`serialization-llm-credentials-label`). The usage and cost of the model are not serialized either.

The items extracted by a decision model are serialized with their document, aspect or concept, with the probabilities.
