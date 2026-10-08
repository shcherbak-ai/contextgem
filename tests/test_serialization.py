#
# ContextGem
#
# Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
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
Module defining tests for serialization.
"""

from __future__ import annotations

import pytest

from contextgem.internal.base.concepts import _Concept
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    JsonObjectConcept,
    JsonObjectExample,
    LabelConcept,
    NumericalConcept,
    RatingConcept,
    StringConcept,
)
from tests.base import BaseTest


class TestSerialization(BaseTest):
    """
    Tests for the serialization, deserialization and cloning of instances.
    """

    # Shared objects passed to the `parametrize` decorators of the tests below.
    # The decorators are evaluated in the class body, where the attributes inherited
    # from the base class are not in scope, so the names are bound here.
    document = BaseTest.document
    document_ua = BaseTest.document_ua
    document_zh = BaseTest.document_zh
    llm_extractor = BaseTest.llm_extractor
    model_group = BaseTest.model_group

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "document",
        [
            document,
            document_ua,
            document_zh,
        ],
    )
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_serialization_and_cloning(
        self, document: Document, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for custom serialization, deserialization, and cloning of application class instances,
        with preservation of the relevant private attributes.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        self.config_llms_for_output_lang(document, llm)

        document = document.clone()  # clone for method-scoped state modification

        # Extraction pipeline serialization
        self.check_instance_serialization_and_cloning(self.extraction_pipeline)
        for aspect in self.extraction_pipeline.aspects:
            self.check_instance_serialization_and_cloning(aspect)
            for concept in aspect.concepts:
                self.check_instance_serialization_and_cloning(concept)
        for concept in self.extraction_pipeline.concepts:
            self.check_instance_serialization_and_cloning(concept)

        # Processed document serialization
        document_aspects = [
            Aspect(
                name="Term and termination",
                description="All clauses addressing term and termination of the contract.",
                concepts=[
                    NumericalConcept(
                        name="Duration",
                        description="Contract duration",
                        numeric_type="int",
                        add_justifications=True,
                        add_references=True,
                        reference_depth="sentences",
                        singular_occurrence=True,
                    ),
                    JsonObjectConcept(
                        name="Duration in years",
                        description="Contract duration in years, if specified.",
                        structure={"years": int | None},
                        examples=[
                            JsonObjectExample(content={"years": 10}),
                            JsonObjectExample(content={"years": None}),
                        ],
                        singular_occurrence=True,
                    ),
                    RatingConcept(
                        name="NDA term adequacy",
                        description="Rate the adequacy of the length of the NDA term, "
                        "from a best practice perspective.",
                        rating_scale=(1, 5),
                        add_justifications=True,
                        justification_depth="balanced",
                        justification_max_sents=5,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Parties",
                description="Parties to the agreement",
                concepts=[
                    StringConcept(
                        name="Contract party name and role",
                        description="Contract party name and role in the contract. "
                        "Exclude information on individual signatories. "
                        "Example: Company ABC (Supplier).",
                        singular_occurrence=False,
                    ),
                    LabelConcept(
                        name="Party types",
                        description="The types of parties involved in the agreement.",
                        labels=[
                            "Individual",
                            "Corporation",
                            "Government entity",
                            "Non-profit organization",
                            "Partnership",
                        ],
                        classification_type="multi_label",
                    ),
                ],
                reference_depth="sentences",
            ),
        ]
        document_images = [
            self.test_img_jpg_invoice,
            self.test_img_jpg_2_invoice,
        ]
        document_concepts = [
            DateConcept(
                name="Start date",
                description="Contract start date",
                add_justifications=True,
                add_references=True,
                reference_depth="sentences",
                singular_occurrence=True,
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                model_role="reasoner",
                add_justifications=True,
                singular_occurrence=True,
            ),
            LabelConcept(
                name="Contract type",
                description="The type(s) of contract represented by this document",
                labels=[
                    "Non-disclosure agreement",
                    "Employment contract",
                    "Service agreement",
                    "Sales contract",
                    "Lease agreement",
                    "Licensing agreement",
                ],
                classification_type="multi_label",
                model_role="reasoner",
                singular_occurrence=True,  # global document classification
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                singular_occurrence=True,
            ),
            NumericalConcept(
                name="Duration in years",
                description="Contract duration in years",
                model_role="reasoner",
                numeric_type="int",
                add_justifications=True,
                singular_occurrence=True,
            ),
            JsonObjectConcept(
                name="Key aspects",
                description="Names and summaries of key aspects covered in the document",
                structure={
                    "aspects_names_and_summaries": [
                        {
                            "aspect_name": str,
                            "aspect_summary": str,
                        }
                    ]  # intentionally overcomplicated structure
                },
                singular_occurrence=True,
            ),
            RatingConcept(
                name="Contract quality",
                description="Rate contract quality from the perspective "
                "of its adherence to best practices in NDA drafting.",
                rating_scale=(1, 10),
                add_justifications=True,
                justification_depth="comprehensive",
                justification_max_sents=10,
                singular_occurrence=True,
            ),
            StringConcept(
                name="Section titles",
                description="Section titles in the contract",
                singular_occurrence=False,
            ),
            # Vision-related
            BooleanConcept(
                name="Invoice check",
                description="Whether the attached images contain an invoice",
                model_role="reasoner",
                modality="image",
                singular_occurrence=True,
            ),
            StringConcept(
                name="Invoice amount",
                description="The total amount of the attached invoice",
                modality="image",
                singular_occurrence=False,  # multiple invoices in the document
            ),
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                modality="image",
                singular_occurrence=False,  # multiple invoices in the document
            ),
            LabelConcept(
                name="Invoice type",
                description="The type of invoice shown in the image",
                labels=[
                    "Commercial invoice",
                    "Proforma invoice",
                    "Tax invoice",
                    "Credit note",
                    "Debit note",
                ],
                classification_type="multi_class",
                modality="image",
                singular_occurrence=False,  # multiple invoices in the document
            ),
        ]
        document.aspects = document_aspects
        document.images = document_images  # ty: ignore[invalid-assignment]
        document.concepts = document_concepts
        document = llm.extract_all(  # ty: ignore[invalid-assignment]
            document,
            max_paragraphs_to_analyze_per_call=30,  # process contract in chunks to test for singular-occurrence concepts
            max_images_to_analyze_per_call=1,  # process invoices 1 by 1, to test for vision concepts in each
        )

        self.log_extracted_items_for_instance(
            document.get_aspect_by_name("Term and termination").get_concept_by_name(
                "NDA term adequacy"
            )
        )
        self.log_extracted_items_for_instance(
            document.get_aspect_by_name("Parties").get_concept_by_name("Party types")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Start date")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Contract quality")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Section titles")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Contract type")
        )
        # Vision-related
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice check")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice amount")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice number")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Key aspects")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice type")
        )

        # Document serialization and deserialization (post-extraction)
        self.check_instance_serialization_and_cloning(document)

        # Aspect serialization and deserialization
        for i, aspect in enumerate(document.aspects):
            self.check_extra_data_in_extracted_items(aspect)  # ty: ignore[invalid-argument-type]
            self.check_instance_serialization_and_cloning(aspect)

            for extracted_item in aspect.extracted_items:
                self.check_instance_serialization_and_cloning(extracted_item)
                for para in extracted_item.reference_paragraphs:
                    self.check_instance_serialization_and_cloning(para)
                    for sentence in para.sentences:
                        self.check_instance_serialization_and_cloning(sentence)

            for concept in aspect.concepts:
                # Aspect concept serialization and deserialization
                aspect_concept_dict = concept.to_dict()
                if i == 1:
                    with pytest.raises(TypeError):
                        # Test invalid class to reconstruct from dict
                        NumericalConcept.from_dict(aspect_concept_dict)
                self.check_instance_serialization_and_cloning(concept)
                for extracted_item in concept.extracted_items:
                    self.check_instance_serialization_and_cloning(extracted_item)
                    for para in extracted_item.reference_paragraphs:
                        self.check_instance_serialization_and_cloning(para)
                        for sentence in para.sentences:
                            self.check_instance_serialization_and_cloning(sentence)

        # Document concept serialization and deserialization
        for i, concept in enumerate(document.concepts):
            concept_dict = concept.to_dict()
            if i == 1:
                with pytest.raises(TypeError):
                    # Test invalid class to reconstruct from dict
                    NumericalConcept.from_dict(concept_dict)
            self.check_instance_serialization_and_cloning(concept)
            for extracted_item in concept.extracted_items:
                self.check_instance_serialization_and_cloning(extracted_item)
                for para in extracted_item.reference_paragraphs:
                    self.check_instance_serialization_and_cloning(para)
                    for sentence in para.sentences:
                        self.check_instance_serialization_and_cloning(sentence)

        # Document image serialization and deserialization
        for _i, image in enumerate(document.images):
            self.check_instance_serialization_and_cloning(image)

        # Check with extraction pipeline assignment
        with pytest.raises(RuntimeError):
            document.assign_pipeline(self.extraction_pipeline)
        document.remove_all_instances()
        document.assign_pipeline(self.extraction_pipeline)
        llm.extract_all(document)
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice number check")
        )
        self.check_instance_container_states(
            original_container=self.extraction_pipeline.aspects,  # ty: ignore[invalid-argument-type]
            assigned_container=document.aspects,  # ty: ignore[invalid-argument-type]
            assigned_instance_class=Aspect,
        )
        self.check_instance_container_states(
            original_container=list(self.extraction_pipeline.concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )
        self.check_instance_serialization_and_cloning(document)
        self.check_instance_serialization_and_cloning(self.extraction_pipeline)

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm)

        # Check serialization of an LLM with fallback
        self._check_deserialized_llm_config_eq(self.invalid_llm_with_valid_fallback)

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)
