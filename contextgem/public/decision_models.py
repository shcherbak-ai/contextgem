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
no text, which makes it faster and cheaper than an LLM for the concepts whose values
come from a closed set of options, and lets it report how sure it is of each value.

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
    generates no text. It extracts the concepts whose values come from a closed set
    of options: ``BooleanConcept``, ``LabelConcept`` and ``RatingConcept``. Each
    extracted item has the probabilities of all the options in its ``probabilities``
    attribute. A decision model can be used on its own or, with a role assigned,
    as a part of a model group, next to the LLMs that extract everything else.

    :ivar model: Model identifier in format {model_provider}/{model_name}, e.g.
        ``"typesafe/jev-latest"``. The model has to be served by one of the providers
        of decision models that ``litellm`` supports.
    :vartype model: str
    :ivar api_key: API key for authentication. Not required for self-hosted models.
        Not included in the serialized model data, i.e. to be set again after
        deserialization.
    :vartype api_key: str | None
    :ivar api_base: Base URL of the API endpoint. Not included in the serialized
        model data, i.e. to be set again after deserialization.
    :vartype api_base: str | None
    :ivar role: Free-form role label of the model (up to 64 printable characters), used
        for routing in a ``DocumentModelGroup``: the model extracts the concepts with
        the same ``model_role``. Defaults to None, in which case the model extracts
        the concepts without ``model_role``. Ignored when the model is used on its own,
        as it then extracts all the concepts it is given.
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
        set together with ``pricing_details``. Not supported for self-hosted models
        (``strands_decider/``) and for ``openai/`` models, for which genai-prices has
        only the prices of the chat requests: set ``pricing_details`` for them
        instead. Defaults to ``False``.
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
        param. A fallback LLM extracts the same concepts with its prompts, i.e. its
        items have no probabilities. Defaults to an empty list.
    :vartype fallback_models: list[DocumentDecisionModel | DocumentLLM]
    :ivar async_limiter: Controls frequency of async API requests for concurrent tasks.
        Defaults to allowing 10 acquisitions per 10-second period to prevent rate limit issues.
        Requests to a fallback model are controlled by the limiter of that fallback model.
        See https://github.com/mjpieters/aiolimiter for configuration details.
    :vartype async_limiter: AsyncLimiter

    Note:

        - What a decision model extracts
            Only ``BooleanConcept``, ``LabelConcept`` and ``RatingConcept``, from text
            (``modality="text"``), without justifications (``add_justifications=False``)
            and without references (``add_references=False``), from a document or from
            an aspect. A ``ValueError`` is raised before any request is sent if
            the model is to extract anything else, such as an aspect or a concept
            of another type. In a model group, route such aspects and concepts to
            an LLM.

        - Extracted items
            A concept gets at most one item per document or aspect, whatever its
            ``singular_occurrence`` is. A ``BooleanConcept`` always gets an item
            (``True`` if the probability of "yes" is 0.5 or more). A ``LabelConcept``
            with ``classification_type="multi_label"`` gets an item with the labels
            whose probability is 0.5 or more, and no item if there are none.
            A ``RatingConcept`` gets the most probable integer of its scale.
            The probabilities are what the model reports and are not a guarantee
            of correctness.

        - Requests
            The questions of all the concepts extracted in one call are sent in
            one request, with the text of the document or of the paragraphs of
            the aspect. The number of concepts per call is set with
            ``max_items_per_call``. With ``use_concurrency=True``, each concept is
            extracted in its own request by default, i.e. the text is sent, and
            is paid for, once per concept: set ``max_items_per_call`` to send
            the text fewer times.

        - Length of the text
            The text of the document, or of the paragraphs of the aspect, is sent
            in one request, i.e. it has to fit into the context window of the model:
            ``max_paragraphs_to_analyze_per_call`` is not applied to a decision model.
            For a long document, extract the concepts from aspects.

        - Limits
            The limits on the number of options are those of ``litellm`` and of
            the model provider. For example, ``litellm`` rejects a rating scale
            with more than 10 levels.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/decision_models/def_decision_model.py
            :language: python
            :caption: Decision model definition
    """

    pass
