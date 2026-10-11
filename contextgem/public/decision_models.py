#
# ContextGem
#
# Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Module for handling processing logic using decision models.

This module provides the class for using a decision model in document processing
workflows. A decision model takes a text and a set of closed questions (yes/no, pick
one option, rate on a scale) and returns the probabilities of the answers. It generates
no text. It is used for what can be put as such questions: which paragraphs or
sentences of a document belong to an aspect, and the concepts whose values come from
a closed set of options. The probabilities are the model's own estimate of each answer.

The requests are sent through the litellm library, which supports several providers
of decision models. The module covers the configuration of a model, the requests to
its API, and the tracking of token usage and costs.
"""

from __future__ import annotations

from contextgem.internal.base.decision_models import _DocumentDecisionModel
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_DocumentDecisionModel)  # ty: ignore[too-many-positional-arguments]
class DocumentDecisionModel(_DocumentDecisionModel):
    """
    Handles processing documents with a specific decision model.

    A decision model answers closed questions on a text with probabilities and
    generates no text. It extracts aspects, by deciding on each paragraph whether
    the paragraph belongs to an aspect and, for an aspect with
    ``reference_depth="sentences"``, on each sentence of such a paragraph whether
    the sentence does, and the concepts whose values come from
    a closed set of options: ``BooleanConcept``, ``LabelConcept`` and
    ``RatingConcept``. Each extracted item has the probabilities of all the options
    in its ``probabilities`` attribute. A decision model can be used on its own or,
    with a role assigned, as a part of a model group, next to the LLMs that extract
    everything else.

    :ivar model: Model identifier in format {model_provider}/{model_name}, e.g.
        ``"typesafe/jev-latest"``. The model has to be served by one of the providers
        of decision models that ``litellm`` supports: a model of another provider
        is rejected when the decision model is created. For a model deployed in
        Microsoft Foundry, the name of the deployment takes the place of the model
        name, i.e. ``"azure_ai/<deployment_name>"``.
    :vartype model: str
    :ivar api_key: API key for authentication. Not required for self-hosted models.
        Not included in the serialized model data, i.e. to be set again after
        deserialization. Not shown when the model is printed, nor in the text of
        a validation error of the model, which names the invalid parameter and
        the reason without showing the values passed. The values are still returned
        by the ``errors()`` and ``json()`` methods of the error, unless
        ``include_input=False`` is passed, with ``'**********'`` in place of a key
        that is passed as a value, e.g. a rejected key or the key among
        the parameters that a model is created with, and in place of the value of
        any parameter that a model does not have, as it may be the key passed under
        a misspelled name. A model that is among the values is returned by
        ``errors()`` as it is, i.e. with the key as its attribute, and is written
        out by ``json()`` without its key. A key that is passed as the value of
        another parameter, e.g. of ``model``, is returned as any other value.
    :vartype api_key: str | None
    :ivar api_base: Base URL of the API endpoint, e.g. the address of a self-hosted
        server, or the endpoint of the Microsoft Foundry resource that the model is
        deployed in. Not included in the serialized model data, i.e. to be set
        again after deserialization.
    :vartype api_base: str | None
    :ivar role: Free-form role label of the model (up to 64 printable characters), used
        for routing in a ``DocumentModelGroup``: the model extracts the aspects/concepts
        with the same ``model_role``. Defaults to None, in which case the model extracts
        the aspects/concepts without ``model_role``. Ignored when the model is used on
        its own, as it then extracts all the aspects/concepts it is given.
    :vartype role: str | None
    :ivar num_retries_failed_request: Number of retries when a request fails with
        a transient error (request timeout, rate limit, server or connection error).
        Retries are performed with exponential backoff and jitter. Non-transient errors
        (e.g. invalid request, failed authentication) are not retried. Defaults to 3.
    :vartype num_retries_failed_request: int
    :ivar timeout: Timeout in seconds for API calls. Defaults to 600 seconds.
    :vartype timeout: int
    :ivar pricing_details: ModelPricing object with pricing details for cost calculation.
        Decision models are usually priced by the input tokens only, in which case
        the price of the output tokens is to be set to ``0.0``. Defaults to None.
    :vartype pricing_details: ModelPricing | None
    :ivar auto_pricing: Enable automatic cost calculation using genai-prices. Cannot be
        set together with ``pricing_details``. Not supported for a model for which
        genai-prices has the prices of a text-generating model, i.e. a price of
        output tokens, which are the prices of the chat requests to the model
        (e.g. ``gpt-6-luna``), and for self-hosted models (``strands_decider/``):
        set ``pricing_details`` for them instead. For a model that genai-prices
        has no prices for, e.g. a model deployed in Microsoft Foundry, a warning
        is issued and the costs stay at zero: ``pricing_details`` is to be set
        instead for such a model too. Defaults to ``False``.
    :vartype auto_pricing: bool
    :ivar auto_pricing_refresh: Whether genai-prices should auto-refresh its cached
        pricing data, e.g. to price a recently released model that the bundled
        pricing data does not include yet. Defaults to ``False``.
    :vartype auto_pricing_refresh: bool
    :ivar is_fallback: Indicates whether the model is a fallback model. Defaults to False.
    :vartype is_fallback: bool
    :ivar fallback_models: Models to use as fallbacks if the current one fails, tried
        in the given order until one of them succeeds: decision models, LLMs, or both.
        Each must have ``is_fallback=True`` and the same role as the current model, and
        must differ from the current model and from each other in at least one config
        param. A fallback LLM extracts the same aspects/concepts with its prompts,
        i.e. its items have no probabilities, and is asked again when it returns
        invalid data, up to its ``max_retries_invalid_data`` times, while a fallback
        decision model is asked once. Defaults to an empty list.
    :vartype fallback_models: list[DocumentDecisionModel | DocumentLLM]
    :ivar decision_threshold: The lowest probability of "yes" (above 0, up to 1) with
        which a yes/no question is taken as answered "yes": a paragraph or a sentence
        belongs to an aspect, a ``BooleanConcept`` is ``True``, a label of
        a ``LabelConcept`` with
        ``classification_type="multi_label"`` is selected. A lower threshold extracts
        more, a higher one extracts only what the model is surer of. Applies unless
        the aspect or the concept has a ``decision_threshold`` of its own. Does not
        apply to a ``LabelConcept`` with ``classification_type="multi_class"`` and to
        a ``RatingConcept``, for which the most probable option is taken. Does not
        change the ``probabilities`` of the extracted items. Defaults to 0.5.
    :vartype decision_threshold: float
    :ivar async_limiter: Controls frequency of async API requests for concurrent tasks.
        Defaults to allowing 10 acquisitions per 10-second period to prevent rate limit issues.
        With concurrency enabled, the requests for the paragraphs of a document, which
        aspects are extracted with, are paced by the limiter: consider a higher rate
        for long documents, if the rate limits of the model provider allow for it.
        The rate applies to all the requests to the model that are sent concurrently,
        i.e. the extraction calls that run at the same time share it.
        Requests to a fallback model are controlled by the limiter of that fallback model.
        See https://github.com/mjpieters/aiolimiter for configuration details.
    :vartype async_limiter: AsyncLimiter

    Note:

        - What a decision model extracts
            Aspects, including sub-aspects, with paragraphs or sentences as their
            extracted items (``reference_depth="paragraphs"`` or
            ``reference_depth="sentences"``), and ``BooleanConcept``, ``LabelConcept``
            and ``RatingConcept``, from text (``modality="text"``) and without
            references (``add_references=False``), from a document or from an aspect.
            Both without justifications (``add_justifications=False``).
            A concept with ``singular_occurrence=True`` is extracted unless
            ``max_paragraphs_to_analyze_per_call`` is passed to the extraction
            method, as it would get an item for each part of the text (see
            "Length of the text").
            A ``ValueError`` is raised before any request is sent if the model is to
            extract anything else, such as a concept of another type. In a model
            group, route such aspects and concepts to an LLM.

        - Extracted items of aspects
            Each paragraph of the document is judged on its own, i.e. without
            the other paragraphs. A paragraph that has ``additional_context`` is
            sent with it: the text of its request has the paragraph and its
            additional context, each with a label, and the question tells that
            the paragraph is what is to be judged. A paragraph belongs
            to an aspect if the probability of that is the decision threshold
            (0.5 by default) or more. Each such
            paragraph becomes an item of the aspect, as when the aspect is extracted
            by an LLM: the value of the item is the text of the paragraph, and
            the paragraph is its reference paragraph. The ``probabilities`` of
            the item are those of the paragraph belonging (``True``) and not
            belonging (``False``) to the aspect. An aspect that no paragraph belongs
            to has no items. The sub-aspects of an aspect are extracted from
            the paragraphs of that aspect.

        - Extracted items of aspects with sentences as the items
            An aspect with ``reference_depth="sentences"`` is extracted in two
            stages. First, each paragraph is judged as above. Then, each sentence
            of each paragraph that belongs to the aspect is judged: the text of
            its request has the sentence and, as its context, the paragraph that
            it is from, each with a label, with the ``additional_context`` of
            the sentence and of the paragraph, if they have any. A sentence
            belongs to the aspect if the probability of that is the decision
            threshold or more, i.e. the same threshold applies at both stages.
            Each such sentence becomes an item of the aspect, as when the aspect
            is extracted by an LLM: the value of the item is the text of
            the sentence, the sentence is its reference sentence, and
            the paragraph is its reference paragraph. The ``probabilities`` of
            the item are those of the sentence. A paragraph that belongs to
            the aspect while none of its sentences does gives no items.

        - Extracted items of concepts
            A concept gets at most one item per document or aspect, with
            ``singular_occurrence=True`` and without it, or, if the text is sent
            in parts (see "Length of the text"), at most one item for each part,
            in the order of the parts. The answers on different parts are not
            merged, i.e. combining such items is up to the caller, and an item
            does not tell which part it comes from.
            A ``BooleanConcept`` always gets an item
            (``True`` if the probability of "yes" is the decision threshold or more,
            0.5 by default). A ``LabelConcept`` with
            ``classification_type="multi_label"`` gets an item with the labels
            whose probability is the decision threshold or more, and no item if
            there are none.
            A ``RatingConcept`` gets the most probable integer of its scale.
            The probabilities are what the model reports and are not a guarantee
            of correctness.

        - Requests
            Concepts: the questions of all the concepts extracted in one call are
            sent in one request, with the text of the document or of the paragraphs
            of the aspect, or in one request for each part of that text, if it is
            sent in parts. Aspects: one request is sent for each paragraph of
            the document, with the text of the paragraph and one question for each
            aspect extracted in the call. For the aspects with
            ``reference_depth="sentences"``, one more request is then sent for each
            sentence of each paragraph that belongs to such an aspect, with one
            question for each of such aspects that the paragraph belongs to:
            the number of requests is that of the paragraphs plus that of
            the sentences in the paragraphs that belong. Aspects with different
            ``reference_depth`` are extracted in separate calls. The number of
            aspects/concepts per call is
            set with ``max_items_per_call``. With ``use_concurrency=True``, each
            aspect/concept is extracted in its own call by default, i.e. each text
            is sent, and is paid for, once per aspect/concept: set
            ``max_items_per_call`` to send the texts fewer times. The settings
            that a decision model does not use (``justification_depth``,
            ``justification_max_sents`` and, for concepts, ``reference_depth``)
            do not split the aspects/concepts into separate calls. The requests for
            the paragraphs, then those for the sentences, and those for the parts
            of a text, are sent one after another, or concurrently with
            ``use_concurrency=True``, paced by the ``async_limiter`` of the model.
            A response that cannot be used, e.g. with a refused answer to
            a question of a concept, is not requested again: the call fails with
            ``ModelExtractionError``, unless a fallback model takes over.
            A paragraph or a sentence that the model refuses to answer on for
            an aspect is not included in that aspect, and a warning names
            the aspect and what was refused: the call fails only if the model
            refuses to answer on every paragraph for an aspect.

        - Length of the text
            Concepts: by default, the text of the document, or of the paragraphs of
            the aspect, is sent in one request, i.e. it has to fit into the context
            window of the model. For a longer text, pass
            ``max_paragraphs_to_analyze_per_call`` to the extraction method:
            a text with more paragraphs is then split into consecutive parts of at
            most that many paragraphs, as for an LLM, and one request is sent for
            each part. Each concept then gets an item for each part, with
            the probabilities of the answers on that part. A decision model
            answers on every part, i.e. it cannot stop at the first part that
            gives an item: with ``max_paragraphs_to_analyze_per_call`` above 0,
            a concept with ``singular_occurrence=True`` is rejected with
            a ``ValueError`` before any request is sent, whatever the number of
            the paragraphs of the text. Extracting the concepts from aspects
            also shortens the text. Aspects: each paragraph is sent in a request
            of its own, whatever the length of the document, i.e.
            ``max_paragraphs_to_analyze_per_call`` changes nothing for them.

        - Limits
            The limits on the number of options are those of ``litellm`` and of
            the model provider. A concept with more options than ``litellm``
            accepts in a question, e.g. a rating scale with more than 10 levels,
            is rejected with a ``ValueError`` before any request is sent, as
            the other concepts that a decision model cannot extract. More
            questions on a text than ``litellm`` accepts in a request (128) are
            sent in several requests on that text.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/decision_models/def_decision_model.py
            :language: python
            :caption: Decision model definition
    """

    pass
