# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling document processing pipelines.

This module provides the ExtractionPipeline class, which represents a reusable collection
of pre-defined aspects and concepts that can be assigned to documents. Pipelines enable
standardized document analysis by packaging common extraction patterns into reusable units.

Pipelines serve as templates for document processing, allowing consistent application of
the same analysis approach across multiple documents. They encapsulate both the structural
organization (aspects) and the specific information to extract (concepts) in a single,
assignable object.
"""

from __future__ import annotations

from contextgem.internal.base.pipelines import _ExtractionPipeline
from contextgem.internal.decorators import _expose_in_registry


@_expose_in_registry(additional_key=_ExtractionPipeline)
class ExtractionPipeline(_ExtractionPipeline):
    """
    Represents a reusable collection of predefined aspects and concepts for document analysis.

    Extraction pipelines serve as templates that can be assigned to multiple documents,
    ensuring consistent application of the same analysis criteria. They package common
    extraction patterns into reusable units, allowing for standardized document processing.

    :ivar aspects: A list of aspects to extract from documents. Aspects represent structural
        categories of information. Defaults to an empty list.
    :vartype aspects: list[_Aspect]
    :ivar concepts: A list of concepts to identify within documents. Concepts represent
        specific information elements to extract. Defaults to an empty list.
        The pipeline holds copies of the aspects and concepts given, and
        ``Document.assign_pipeline()`` gives the document copies of its own: read
        the extracted items from the document's aspects and concepts.
    :vartype concepts: list[_Concept]
    :ivar model_roles: The ``model_role`` values of the pipeline's aspects and concepts,
        and of the sub-aspects and concepts of its aspects, with None for those without
        a role. Read-only.
    :vartype model_roles: set[str | None]
    :ivar custom_data: A dictionary for data of your own about the pipeline, e.g. the ID
        that your application has for it. Must be JSON-serializable. Not sent to the models,
        and kept when the pipeline is serialized or cloned. Defaults to an empty dict.
    :vartype custom_data: dict[str, Any]
    :ivar unique_id: The unique ID of the pipeline, a ULID assigned when the pipeline is
        created. Read-only. Kept when the pipeline is serialized or cloned, i.e. a restored
        or cloned pipeline has the ID of the original.
    :vartype unique_id: str

    Note:
        A pipeline is a reusable configuration of extraction steps. You can use the same pipeline
        to extract data from multiple documents.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/pipelines/def_pipeline.py
            :language: python
            :caption: Extraction pipeline definition

        .. The example above is not in the installed package. It is shown at
           https://contextgem.shcherbak.ai/api/pipelines/index.html.md
    """
