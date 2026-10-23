# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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


@_expose_in_registry(additional_key=_DocumentDecisionModel)
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
        set together with ``pricing_details``. For an OpenAI model (``openai/``),
        the prices of the decision requests are used, which genai-prices has apart
        from the prices of the chat requests to the model. Not supported for
        a model for which genai-prices has the prices of a text-generating model,
        i.e. a price of output tokens, which are the prices of the chat requests
        to the model (e.g. ``openrouter/openai/gpt-6-luna``), and for self-hosted
        models (``strands_decider/``): set ``pricing_details`` for them instead.
        For a model that genai-prices has no prices for, e.g. a model deployed in
        Microsoft Foundry, a warning is issued and the costs stay at zero:
        ``pricing_details`` is to be set instead for such a model too. Defaults to
        ``False``.
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
    :ivar is_group: Whether the object is a model group: always False for a decision
        model. Read-only.
    :vartype is_group: bool
    :ivar list_roles: The role of the decision model, as a list of one element:
        ``[role]``, with None for a model without a role. Read-only.
    :vartype list_roles: list[str | None]

    Note:
        For what a decision model extracts, how its requests are formed, how
        the extracted items get their values and probabilities, and the limits,
        see https://contextgem.shcherbak.ai/decision_models/decision_models/

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/decision_models/def_decision_model.py
            :language: python
            :caption: Decision model definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/decision_models/index.html.md
    """
