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
- aspects, by deciding for each paragraph whether it belongs to the aspect.

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
     - Yes, one paragraph at a time, with paragraphs as the extracted items
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
   * - Self-hosted
     - ``strands_decider/<model_name>``
     - ``api_base``. No API key is needed. The requests are sent to ``POST <api_base>/v1/systemone``.

The model names above are the ones in the model list of this ``litellm`` version.

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_init.py
   :language: python
   :caption: Creating decision models

What to know about the providers:

- **The provider and the model name are not checked when a decision model is created.** A name that LiteLLM cannot serve, a missing API key and a missing API base all fail at the first request, with ``ModelAPIError`` (see :ref:`decision-models-errors-label`).
- **API keys in the environment.** If ``api_key`` is not passed, LiteLLM reads the key from the environment variable of the provider, if it is set, e.g. ``TYPESAFE_API_KEY`` or ``OPENAI_API_KEY``.
- **What is tested.** The tests of ContextGem send requests to TypeSafe (``typesafe/jev-latest``) and to the Decisions API of OpenAI (``openai/gpt-6-luna``). The other providers are reached through the same LiteLLM call and are not tested with ContextGem.

.. warning::
   **Ollama and LM Studio are not verified.** LiteLLM has no decision provider under the ``ollama/``, ``ollama_chat/`` and ``lm_studio/`` prefixes, which are for LLMs. A decision model can be created with such a prefix, but its first request fails before it is sent, with ``ModelAPIError`` (*Unknown Decisions provider 'ollama'*). A local server is reached with the ``strands_decider/`` prefix and its ``api_base``, if it accepts ``POST <api_base>/v1/systemone``. Whether Ollama or LM Studio serve this endpoint has not been verified with ContextGem.

ContextGem sends only text to a decision model. Concepts with the ``"image"`` or ``"multimodal"`` modality are not supported, whichever provider is used.


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
     - Enable automatic cost calculation using ``genai-prices``. Mutually exclusive with ``pricing_details``. Rejected for the decision models whose requests cannot be priced with it, e.g. ``openai/`` and self-hosted (``strands_decider/``) ones. See :ref:`decision-models-cost-label`.
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
   * - ``async_limiter``
     - ``AsyncLimiter``
     - ``AsyncLimiter(10, 10)``
     - Relevant when concurrency is enabled for extraction tasks. Controls the frequency of the requests sent concurrently. Defaults to allowing 10 acquisitions per 10-second period. This matters most for aspect extraction, which sends one request per paragraph (see :ref:`decision-models-aspects-label`). See `aiolimiter documentation <https://github.com/mjpieters/aiolimiter>`_ for AsyncLimiter configuration details.

A decision model has none of the other parameters of :class:`~contextgem.public.llms.DocumentLLM`, e.g. those that control the generation of text or the chat (``system_message``, ``output_language``, ``temperature``, ``top_p``, ``seed``, ``max_tokens``, ``reasoning_effort``, ``tools``), or ``max_retries_invalid_data``. Passing one of them raises a validation error.


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
     - With ``reference_depth="paragraphs"`` (default) and without justifications
     - ``reference_depth="sentences"``; ``add_justifications=True``
   * - :class:`~contextgem.public.concepts.BooleanConcept`, :class:`~contextgem.public.concepts.LabelConcept` (both classification types), :class:`~contextgem.public.concepts.RatingConcept`
     - With ``modality="text"`` (default), without justifications and without references. The concept can be attached to a document or to an aspect.
     - ``modality="image"`` or ``modality="multimodal"``; ``add_justifications=True``; ``add_references=True``
   * - :class:`~contextgem.public.concepts.StringConcept`, :class:`~contextgem.public.concepts.NumericalConcept`, :class:`~contextgem.public.concepts.DateConcept`, :class:`~contextgem.public.concepts.JsonObjectConcept`
     - Not supported
     - Always

The number of the labels of a single-label concept and the length of a rating scale are limited as well (see :ref:`decision-models-limits-label`).

The error names the model and each aspect or concept that cannot be extracted, with the reasons:

.. code-block:: text
   :caption: The error for a concept that a decision model cannot extract

   ValueError: Decision model `typesafe/jev-latest` cannot extract: concept `Fee` (type `StringConcept` is not supported). A decision model extracts aspects with the `paragraphs` reference depth, and Boolean, label and rating concepts from text without references, both without justifications. Route the other aspects and concepts to an LLM, e.g. in a model group.

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
     - ``True`` if the probability of "yes" is 0.5 or more, else ``False``
   * - :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_class"``
     - One question with the labels as the options, of which one is to be picked
     - A list with the most probable label, e.g. ``["Written"]``
   * - :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_label"``
     - One yes/no question per label: does the label apply to the text?
     - A list of the labels with a probability of 0.5 or more, in the order of the labels of the concept. No item is extracted if no label reaches 0.5.
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
- **At most one item is extracted** for a concept from a document or an aspect, whatever ``singular_occurrence`` is set to.

All the questions of the concepts processed in one call are sent in one request, together with the text: the text of the document for document-level concepts, or the text of the paragraphs extracted for the aspect for aspect-level concepts. As for LLMs, ``max_items_per_call`` sets how many concepts are processed in one call (all of them by default, or one with ``use_concurrency=True``).


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
     - The probability ``p`` that the paragraph belongs to the aspect under ``True``, and ``1 - p`` under ``False``
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

The values of the items are set with a fixed threshold of 0.5. To apply a higher threshold, read the probabilities: e.g. take a Boolean concept as confirmed only if ``item.probabilities[True] >= 0.9``.

What is not extracted has no probabilities to read. A multi-label concept none of whose labels reaches 0.5 has no extracted item, and a paragraph with a probability below 0.5 is not among the extracted items of the aspect. A threshold below 0.5 can therefore not be applied to them afterwards. The answers of the model to each request, including these, are recorded as JSON in the usage of the model (see :ref:`decision-models-cost-label`).

``probabilities`` is ``None`` for every item extracted by an LLM, including an LLM that is a fallback of a decision model. The probabilities are kept when a document is serialized (see :doc:`../serialization`).

See :ref:`decision-models-trust-label` for what a probability does and does not tell.


.. _decision-models-aspects-label:

📄 Extracting Aspects
-----------------------

A decision model extracts aspects one paragraph at a time:

- **One request per paragraph.** A request has the text of one paragraph and, for each aspect processed in the call, one yes/no question: is the paragraph relevant to the aspect? The question is built from the ``name`` and the ``description`` of the aspect.
- **A paragraph is judged on its own text.** The model is not sent the neighbouring paragraphs, the rest of the document, or the ``additional_context`` of the paragraph, whereas an LLM is sent the paragraphs together. A paragraph that can be understood only in its context, e.g. *"The same applies to the Supplier."*, can therefore be judged differently than by an LLM.
- **A paragraph with a probability of 0.5 or more becomes an extracted item** of the aspect. As for LLMs, the value of the item is the text of the paragraph, and ``reference_paragraphs`` of the item has the paragraph. A paragraph can belong to several aspects.
- **An aspect that no paragraph belongs to has no extracted items.** Its sub-aspects and concepts are then not extracted, as for LLMs.

The sub-aspects of an aspect are extracted in the same way from the paragraphs extracted for that aspect, with one request for each of these paragraphs. The concepts of an aspect are decided on the text of the paragraphs extracted for it.

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_aspects.py
   :language: python
   :caption: Extracting aspects and their concepts with a decision model

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
   **When you enable concurrency, also set the max items per call.** As for LLMs, each aspect and each concept is processed in a call of its own when ``use_concurrency=True`` is passed and ``max_items_per_call`` is not set. With a decision model, each text is then sent, and billed, once per aspect or concept: 3 aspects and 100 paragraphs take 300 requests instead of 100. To send each text once, with the questions of all the aspects or concepts, set ``max_items_per_call`` to the number of the aspects or concepts, or more.

``max_paragraphs_to_analyze_per_call`` is not applied to a decision model: for aspects, each paragraph is sent in a request of its own in any case, and for concepts, the text is sent as a whole (see :ref:`decision-models-limits-label`).

The length of the document does not limit aspect extraction, as each request has one paragraph only.


.. _decision-models-model-group-label:

🤖🤖 Using a Decision Model in a Model Group
----------------------------------------------

In a :class:`~contextgem.public.models.DocumentModelGroup`, each aspect and concept is extracted by the model whose ``role`` equals its ``model_role`` (see :ref:`model-roles-label`). Give the decision model a role, and assign the same ``model_role`` to the aspects and concepts it is to extract. The LLM in the group extracts the rest, e.g. the concepts with open values:

.. literalinclude:: ../../../dev/usage_examples/docs/decision_models/decision_model_group.py
   :language: python
   :caption: A decision model and an LLM in a model group

The routing is yours to declare: ContextGem does not pick a model for an aspect or a concept. If an aspect or a concept that a decision model cannot extract is routed to it, ``ValueError`` is raised before any request is sent (see :ref:`decision-models-supported-label`).

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

- ``auto_pricing=True`` works for the models whose prices of decision requests ``genai-prices`` has. With the pinned ``genai-prices`` version (0.1.9), these are the TypeSafe models listed above.
- For a model that ``genai-prices`` has no price for, a warning is issued when the model is created with ``auto_pricing=True`` and when its usage is recorded, and the costs stay at zero. With ``genai-prices`` 0.1.9, this is the case for the decision models of Perplexity, OpenRouter and Cloudflare listed above: set ``pricing_details`` to get the costs of these models.
- ``auto_pricing=True`` is rejected with a validation error (a ``ValueError``) for a decision model whose requests cannot be priced with it, when the model is created and when ``auto_pricing`` or ``model`` is assigned. This is the case e.g. for ``openai/`` decision models: ``genai-prices`` has only the chat prices of ``gpt-6-luna``, which differ from its prices on the Decisions API, so the costs would be calculated with the wrong prices without any notice. It is also the case for self-hosted models (``strands_decider/``), which have no prices. Set ``pricing_details`` for such a model instead.
- With ``pricing_details``, :class:`~contextgem.public.data_models.ModelPricing` requires both ``input_per_1m_tokens`` and ``output_per_1m_tokens``. Set ``output_per_1m_tokens=0.0`` for a provider that does not charge for output.

As of October 2026, TypeSafe lists $0.042 per 1M input tokens for ``jev-1.13.0`` (`TypeSafe models <https://docs.typesafe.ai/models>`_), and OpenAI lists $0.10 per 1M input tokens for ``gpt-6-luna`` on the Decisions API (`OpenAI Decisions guide <https://developers.openai.com/api/docs/guides/decisions>`_). Neither charges for output tokens. Check the pricing page of your provider for the current prices.

**What a request costs.** The text is sent, and billed as input tokens, once per request: the text of the document or of the paragraphs of the aspect for each call of concepts, and the text of the paragraph for each request of aspect extraction. The fewer requests a text is sent in, the less it costs:

- With ``use_concurrency=True``, set ``max_items_per_call``, as each aspect and each concept is otherwise processed in a call of its own (see :ref:`decision-models-aspects-label`).
- As for LLMs, only the aspects or concepts with the same extraction parameters are processed in one call. Aspects and concepts that differ in ``justification_depth`` or ``justification_max_sents``, and concepts that differ in ``reference_depth``, are therefore sent in separate requests, although a decision model does not use these parameters. Leave them at their defaults for the aspects and concepts of a decision model.
- More than 128 questions on a text are sent in several requests (see :ref:`decision-models-limits-label`).


.. _decision-models-limits-label:

📏 Limits
-----------

**Limits of litellm.** ``litellm`` accepts a request only within the following limits:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Limit
     - What it means for the extraction
   * - Up to 128 questions per request
     - Not a limit for the extraction: ContextGem sends the questions over the cap in further requests on the same text. A Boolean, single-label or rating concept takes one question, and a multi-label concept takes one question per label, so e.g. a multi-label concept with 130 labels takes two requests. The text is sent, and billed, once per request. In aspect extraction, each aspect of the call takes one question on each paragraph.
   * - 2 to 255 options of a question with one option to pick
     - A :class:`~contextgem.public.concepts.LabelConcept` with ``classification_type="multi_class"`` can have up to 255 labels. A multi-label concept is not affected, as each of its labels is a question of its own.
   * - 2 to 10 levels of a rating question
     - The rating scale of a :class:`~contextgem.public.concepts.RatingConcept` can have up to 10 levels: ``rating_scale=(1, 10)`` works, while ``rating_scale=(0, 10)`` has 11 levels and does not.

A concept over the limit on the options or on the levels cannot be extracted with a decision model: the extraction fails, and no request with its question is sent to the provider. Use an LLM for such a concept, e.g. by routing it to an LLM in a model group.

**The text must fit in one request.** A document-level concept is decided on the text of the whole document, which is sent in one request: the text is not split into parts, and its length is not checked before the request is sent. A request that the provider rejects for its length fails as any other rejected request. For Jev 1.13, TypeSafe documents a limit of 32k tokens for the text together with the longest question, and of 64k tokens per request (`TypeSafe models <https://docs.typesafe.ai/models>`_, as of October 2026). For a longer document, attach the concepts to aspects: the aspects are extracted paragraph by paragraph, and the concepts of an aspect are decided on the paragraphs of the aspect only, which also have to fit in one request.

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
     - When an extraction method is called, before any request is sent: the decision model, or a decision model in the fallback chain, cannot extract an aspect or a concept that it is given (see :ref:`decision-models-supported-label`).
   * - ``ModelAPIError``
     - A request failed for good: with a non-transient error, or with a transient error after ``num_retries_failed_request`` retries. This includes the requests that ``litellm`` rejects before sending them, e.g. for an unknown provider or for a missing API key or API base. The error has the number of retries made (``retry_count``) and the error of the request (``original_error``).
   * - ``ModelExtractionError``
     - A response cannot be used: the model refused to answer a question, an answer is missing, or an answer does not have a valid probability for each of the options offered. The request is not sent again, as the same response is expected.

``ModelAPIError`` and ``ModelExtractionError`` are raised only if there is no fallback model left to try (see :ref:`decision-models-fallbacks-label`).

What to expect when a call fails:

- **One answer that cannot be used fails the whole call.** No aspect or concept of that call gets its items, including those whose answers were valid. The aspects and concepts extracted in other calls keep their items.
- **In aspect extraction, the answers on all the paragraphs are needed.** When a request for a paragraph fails for good, or its response cannot be used, the requests not yet sent are not sent, and no aspect of the call gets its items.
- **The requests made are recorded in the usage**, with their tokens, also when the call fails.
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
