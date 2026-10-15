# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for serialization.
"""

from __future__ import annotations

import json
import re
import types
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal
from unittest.mock import patch

import pytest

from contextgem.internal.base import llms as llms_module
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.items import (
    _BooleanItem,
    _DateItem,
    _FloatItem,
    _IntegerItem,
    _IntegerOrFloatItem,
    _JsonObjectItem,
    _LabelItem,
    _StringItem,
)
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
    Paragraph,
    RatingConcept,
    Sentence,
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

    def test_item_probabilities_serialization(self, tmp_path: Path):
        """
        Tests the serialization, deserialization and cloning of the extracted items
        that have the probabilities of the options: the probabilities are serialized
        as a list of [option, probability] pairs, and are restored exactly, including
        the types and the order of the options.
        """
        # Item class, item value, probabilities, probabilities as serialized to JSON
        test_cases: list[
            tuple[type[_ExtractedItem], Any, dict[bool | int | str, float], str]
        ] = [
            # Yes/no question
            (
                _BooleanItem,
                True,
                {True: 0.83, False: 0.17},
                "[[true, 0.83], [false, 0.17]]",
            ),
            # Item of an aspect, with the options in another order
            (
                _StringItem,
                "Any notice shall be given in writing.",
                {False: 0.09, True: 0.91},
                "[[false, 0.09], [true, 0.91]]",
            ),
            # One label out of several
            (
                _LabelItem,
                ["written"],
                {"written": 0.71, "email": 0.20, "any": 0.06, "not specified": 0.03},
                (
                    '[["written", 0.71], ["email", 0.2], ["any", 0.06], '
                    '["not specified", 0.03]]'
                ),
            ),
            # Several labels, with the probabilities that do not sum to 1
            (
                _LabelItem,
                ["NDA", "Confidential"],
                {"NDA": 0.97, "Confidential": 0.62, "Employment": 0.08},
                '[["NDA", 0.97], ["Confidential", 0.62], ["Employment", 0.08]]',
            ),
            # Labels in other languages
            (
                _LabelItem,
                ["Конфіденційно"],
                {"Конфіденційно": 0.9, "保密": 0.1},
                '[["Конфіденційно", 0.9], ["保密", 0.1]]',
            ),
            # Rating
            (
                _IntegerItem,
                4,
                {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.55, 5: 0.30},
                "[[1, 0.0], [2, 0.0], [3, 0.15], [4, 0.55], [5, 0.3]]",
            ),
            # Rating, with the options in the descending order
            (
                _IntegerItem,
                4,
                {5: 0.30, 4: 0.55, 3: 0.15, 2: 0.0, 1: 0.0},
                "[[5, 0.3], [4, 0.55], [3, 0.15], [2, 0.0], [1, 0.0]]",
            ),
            # Integers that are equal to booleans as dict keys
            (
                _IntegerItem,
                1,
                {0: 0.25, 1: 0.75},
                "[[0, 0.25], [1, 0.75]]",
            ),
            # Options of all types, including the strings that read as other options
            (
                _StringItem,
                "Random string",
                {True: 0.5, "True": 0.3, 2: 0.2, "2": 0.0, 0: 0.0, "false": 0.0},
                (
                    '[[true, 0.5], ["True", 0.3], [2, 0.2], ["2", 0.0], [0, 0.0], '
                    '["false", 0.0]]'
                ),
            ),
            # Probabilities given as integers
            (
                _BooleanItem,
                False,
                {True: 0, False: 1},
                "[[true, 0.0], [false, 1.0]]",
            ),
            # Probabilities that need all the digits of a float
            (
                _LabelItem,
                ["a"],
                {"a": 1 / 3, "b": 0.1 + 0.2, "c": 1e-17, "d": 5e-324},
                (
                    '[["a", 0.3333333333333333], ["b", 0.30000000000000004], '
                    '["c", 1e-17], ["d", 5e-324]]'
                ),
            ),
            # Other item classes, including the one with own serialization
            (
                _FloatItem,
                1.5,
                {True: 0.83, False: 0.17},
                "[[true, 0.83], [false, 0.17]]",
            ),
            (
                _IntegerOrFloatItem,
                1,
                {1: 0.6, 2: 0.4},
                "[[1, 0.6], [2, 0.4]]",
            ),
            (
                _JsonObjectItem,
                {"hello": "world"},
                {"hello": 0.6, "world": 0.4},
                '[["hello", 0.6], ["world", 0.4]]',
            ),
            (
                _DateItem,
                date(2025, 3, 2),
                {True: 0.83, False: 0.17},
                "[[true, 0.83], [false, 0.17]]",
            ),
        ]

        for index, (item_class, value, probabilities, probabilities_json) in enumerate(
            test_cases
        ):
            item = item_class(value=value, probabilities=probabilities)
            self.check_item_probabilities(item, probabilities)

            # To dict: a list of [option, probability] lists, in which the options
            # keep their types and order when converted to JSON
            item_dict = item.to_dict()
            serialized_probabilities = item_dict["probabilities"]
            assert isinstance(serialized_probabilities, list)
            assert len(serialized_probabilities) == len(probabilities)
            assert all(
                type(pair) is list and len(pair) == 2
                for pair in serialized_probabilities
            )
            assert (
                json.dumps(serialized_probabilities, ensure_ascii=False)
                == probabilities_json
            )

            # To JSON
            item_json = item.to_json()
            assert f'"probabilities": {probabilities_json}' in item_json
            assert json.loads(item_json) == item_dict

            # To disk
            disk_path = tmp_path / f"item_{index}.json"
            item.to_disk(disk_path)
            with open(disk_path, encoding="utf-8") as file:
                item_dict_on_disk = json.load(file)
            assert item_dict_on_disk == item_dict
            assert (
                json.dumps(item_dict_on_disk["probabilities"], ensure_ascii=False)
                == probabilities_json
            )

            # From dict, from JSON, from disk, and cloning
            item_dict_before = deepcopy(item_dict)
            for restored_item in (
                item_class.from_dict(item_dict),
                item_class.from_dict(json.loads(item_json)),
                item_class.from_json(item_json),
                item_class.from_disk(disk_path),
                item.clone(),
                item_class.from_json(item_json).clone(),
                item_class.from_json(item_class.from_disk(disk_path).to_json()),
            ):
                assert type(restored_item) is item_class
                assert restored_item is not item
                assert restored_item.probabilities is not item.probabilities
                self.check_item_probabilities(restored_item, probabilities)
                assert restored_item == item
                assert restored_item.value == item.value
                assert restored_item.unique_id == item.unique_id
                # Serialized again, the restored item gives the same data
                assert restored_item.to_json() == item_json

            # The serialized data is not changed when the item is created from it,
            # and is not shared with the item
            assert item_dict == item_dict_before
            assert (
                json.dumps(item_dict["probabilities"], ensure_ascii=False)
                == probabilities_json
            )
            item_dict["probabilities"][0][1] = 0.5
            item_dict["probabilities"].append(["other", 0.0])
            self.check_item_probabilities(item, probabilities)
            assert item.to_json() == item_json

            # Shared checks of the serialization and cloning of an instance
            self.check_instance_serialization_and_cloning(item)

        # No probabilities
        for index, item in enumerate(
            [
                _BooleanItem(value=True),
                _LabelItem(value=["NDA"], probabilities=None),
                _DateItem(value=date(2025, 3, 2)),
            ]
        ):
            item_dict = item.to_dict()
            assert "probabilities" in item_dict
            assert item_dict["probabilities"] is None
            item_json = item.to_json()
            assert '"probabilities": null' in item_json
            disk_path = tmp_path / f"item_without_probabilities_{index}.json"
            item.to_disk(disk_path)
            for restored_item in (
                type(item).from_dict(item_dict),
                type(item).from_json(item_json),
                type(item).from_disk(disk_path),
                item.clone(),
            ):
                assert restored_item.probabilities is None
                assert restored_item == item
                assert restored_item.to_json() == item_json
            self.check_instance_serialization_and_cloning(item)

    @pytest.mark.parametrize(
        "invalid_serialized_probabilities",
        [
            # Not a list of pairs
            {"True": 0.83, "False": 0.17},  # JSON object, which loses the option types
            [True, 0.83],
            "[[true, 0.83], [false, 0.17]]",
            0.83,
            True,
            # Not a pair
            [[True, 0.83, 0.17]],
            [[True]],
            [[]],
            [{"option": True, "probability": 0.83}],
            ["ab"],
            [[True, 0.83], None],
            # Option of an invalid type
            [[1.0, 0.5]],
            [[None, 0.5]],
            [[[True], 0.5]],
            [[{"option": True}, 0.5]],
            # Repeated option, including the options that are equal as dict keys
            [[True, 0.5], [True, 0.5]],
            [["NDA", 0.5], ["NDA", 0.4]],
            [[True, 0.5], [1, 0.5]],
            [[0, 0.5], [False, 0.5]],
            # Probabilities that are invalid for an item
            [],
            [[True, 1.5]],
            [[True, -0.5]],
            [[True, "0.83"]],
            [[True, True]],
            [[True, None]],
            [[True, [0.83]]],
            [[True, float("nan")]],
            [[True, 0.83], [False, float("inf")]],
        ],
    )
    def test_deserialize_item_with_invalid_probabilities(
        self, invalid_serialized_probabilities: Any, tmp_path: Path
    ):
        """
        Tests that an extracted item is not created from the serialized data
        in which the probabilities are invalid.
        """
        item = _BooleanItem(value=True, probabilities={True: 0.83, False: 0.17})
        item_dict = item.to_dict()
        item_dict["probabilities"] = invalid_serialized_probabilities

        # From dict
        with pytest.raises(ValueError, match="probabilities"):
            _BooleanItem.from_dict(item_dict)

        # From JSON (the `json` module writes and reads a NaN and an infinity)
        item_json = json.dumps(item_dict)
        with pytest.raises(ValueError, match="probabilities"):
            _BooleanItem.from_json(item_json)

        # From disk
        disk_path = tmp_path / "item.json"
        disk_path.write_text(item_json, encoding="utf-8")
        with pytest.raises(RuntimeError, match="probabilities"):
            _BooleanItem.from_disk(disk_path)

    def test_deserialize_item_without_probabilities(self, tmp_path: Path):
        """
        Tests that the data serialized before the probabilities were added to
        the extracted items, i.e. the data without the `probabilities` key, is loaded,
        and gives the items without probabilities.
        """
        # Item class, value as serialized, value
        legacy_items: list[tuple[type[_ExtractedItem], Any, Any]] = [
            (_StringItem, "Random string", "Random string"),
            (_BooleanItem, True, True),
            (_IntegerItem, 4, 4),
            (_FloatItem, 1.5, 1.5),
            (_IntegerOrFloatItem, 1, 1),
            (_JsonObjectItem, {"hello": "world"}, {"hello": "world"}),
            (_DateItem, "2025-03-02", date(2025, 3, 2)),
            (_LabelItem, ["NDA"], ["NDA"]),
        ]
        for index, (item_class, serialized_value, value) in enumerate(legacy_items):
            # The data of an item as it was serialized before
            legacy_item_dict = {
                "custom_data": {"source": "legacy"},
                "value": serialized_value,
                "justification": "Random justification",
                "_unique_id": f"01K7ABCDEFGHJKMNPQRSTVWXY{index}",
                "_reference_paragraphs": [],
                "_reference_sentences": [],
                "__class__": item_class.__name__,
            }
            legacy_item_json = json.dumps(legacy_item_dict)
            assert "probabilities" not in legacy_item_json
            disk_path = tmp_path / f"legacy_item_{index}.json"
            disk_path.write_text(legacy_item_json, encoding="utf-8")

            for item in (
                item_class.from_dict(legacy_item_dict),
                item_class.from_json(legacy_item_json),
                item_class.from_disk(disk_path),
            ):
                assert type(item) is item_class
                assert item.probabilities is None
                assert item.value == value
                assert item.justification == "Random justification"
                assert item.custom_data == {"source": "legacy"}
                assert item.unique_id == legacy_item_dict["_unique_id"]
                # Serialized now, the item has the key
                assert item.to_dict() == {**legacy_item_dict, "probabilities": None}
                self.check_instance_serialization_and_cloning(item)
            assert "probabilities" not in legacy_item_dict

        # A document with the extracted items, as it was serialized before
        document = self._get_document_with_extracted_items(with_probabilities=False)
        legacy_document_dict = document.to_dict()
        removed_keys_count = self._remove_probabilities_keys(legacy_document_dict)
        assert removed_keys_count == len(self._get_extracted_items(document)) == 6
        legacy_document_json = json.dumps(legacy_document_dict)
        assert "probabilities" not in legacy_document_json
        disk_path = tmp_path / "legacy_document.json"
        disk_path.write_text(legacy_document_json, encoding="utf-8")
        for restored_document in (
            Document.from_dict(legacy_document_dict),
            Document.from_json(legacy_document_json),
            Document.from_disk(disk_path),
        ):
            assert restored_document == document
            restored_items = self._get_extracted_items(restored_document)
            assert len(restored_items) == 6
            assert all(item.probabilities is None for item in restored_items.values())

    def test_from_json_error_does_not_hold_string(self, tmp_path: Path):
        """
        Tests that the error raised for a string that is not a valid JSON,
        when an instance is created from it with ``from_json()`` or from a file
        with ``from_disk()``, does not hold the string, e.g. the data of an LLM
        that an API key was added to: the error of the JSON parser keeps the whole
        string as its ``doc`` attribute. The rest of the error, i.e. its message
        and the position of what is invalid, is the same.
        """
        api_key = "sk-test-SECRET-0123456789"
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key=api_key)
        other_llm = DocumentLLM(
            model="openai/gpt-6-luna", api_key=api_key, role="other"
        )
        document = Document(raw_text="Services agreement")
        instances_and_fields = [
            (llm, "api_key"),
            (DocumentModelGroup(models=[llm, other_llm]), "note"),
            (document, "note"),
        ]
        for instance, field in instances_and_fields:
            # The data with the key added to it, which is cut off
            data = instance.to_dict()
            data[field] = api_key
            json_string = json.dumps(data, indent=2)
            assert api_key in json_string
            invalid_json_string = json_string[
                : json_string.index(api_key) + len(api_key)
            ]
            assert api_key in invalid_json_string

            # The error that the JSON parser raises holds the string
            with pytest.raises(json.JSONDecodeError) as parser_exc_info:
                json.loads(invalid_json_string)
            parser_error = parser_exc_info.value
            assert parser_error.doc == invalid_json_string

            with pytest.raises(json.JSONDecodeError) as exc_info:
                type(instance).from_json(invalid_json_string)
            error = exc_info.value
            assert error.doc == ""
            for text in (str(error), repr(error), repr(error.args)):
                assert api_key not in text
            # The rest of the error is the same
            assert str(error) == str(parser_error)
            assert (error.msg, error.pos, error.lineno, error.colno) == (
                parser_error.msg,
                parser_error.pos,
                parser_error.lineno,
                parser_error.colno,
            )

            # The error for a file has the same error as its cause
            file_path = tmp_path / f"{type(instance).__name__}.json"
            file_path.write_text(invalid_json_string, encoding="utf-8")
            with pytest.raises(RuntimeError, match="Failed to load") as exc_info:
                type(instance).from_disk(file_path)
            assert api_key not in str(exc_info.value)
            cause = exc_info.value.__cause__
            assert isinstance(cause, json.JSONDecodeError)
            assert cause.doc == ""
            assert str(cause) == str(parser_error)

        # A valid string is restored as before
        assert Document.from_json(document.to_json()) == document

    def test_document_serialization_with_item_probabilities(self, tmp_path: Path):
        """
        Tests that the probabilities of the extracted items of the aspects and
        concepts of a document are kept when the document is serialized,
        deserialized and cloned.
        """
        document = self._get_document_with_extracted_items(with_probabilities=True)
        expected_probabilities = self._document_item_probabilities
        items = self._get_extracted_items(document)
        assert set(items) == set(expected_probabilities)
        for name, item in items.items():
            self.check_item_probabilities(item, expected_probabilities[name])

        # Serialized document
        document_json = document.to_json()
        for probabilities_json in (
            "[[true, 0.91], [false, 0.09]]",
            (
                '[["written", 0.71], ["email", 0.2], ["any", 0.06], '
                '["not specified", 0.03]]'
            ),
            "[[true, 0.83], [false, 0.17]]",
            '[["NDA", 0.97], ["Confidential", 0.62], ["Employment", 0.08]]',
            "[[1, 0.0], [2, 0.0], [3, 0.15], [4, 0.55], [5, 0.3]]",
            "null",
        ):
            assert document_json.count(f'"probabilities": {probabilities_json}') == 1
        assert document_json.count('"probabilities": ') == 6

        # Deserialized and cloned document
        disk_path = tmp_path / "document.json"
        document.to_disk(disk_path)
        for restored_document in (
            Document.from_json(document_json),
            Document.from_dict(document.to_dict()),
            Document.from_dict(json.loads(document_json)),
            Document.from_disk(disk_path),
            document.clone(),
            Document.from_json(Document.from_json(document_json).to_json()),
        ):
            assert restored_document is not document
            assert restored_document == document
            restored_items = self._get_extracted_items(restored_document)
            assert set(restored_items) == set(expected_probabilities)
            for name, restored_item in restored_items.items():
                self.check_item_probabilities(
                    restored_item, expected_probabilities[name]
                )
                assert restored_item == items[name]
                assert restored_item is not items[name]
            # Serialized again, the restored document gives the same data
            assert restored_document.to_json() == document_json

        # A document that differs only in the probabilities of an item is not equal
        document_dict = document.to_dict()
        boolean_item_dict = document_dict["concepts"][0]["_extracted_items"][0]
        assert boolean_item_dict["probabilities"] == [[True, 0.83], [False, 0.17]]
        boolean_item_dict["probabilities"] = [[True, 0.84], [False, 0.16]]
        assert Document.from_dict(document_dict) != document
        boolean_item_dict["probabilities"] = [[True, 0.83], [False, 0.17]]
        assert Document.from_dict(document_dict) == document

        # Shared checks of the serialization and cloning of an instance
        self.check_instance_serialization_and_cloning(document)
        for aspect in document.aspects:
            self.check_instance_serialization_and_cloning(aspect)
            for concept in aspect.concepts:
                self.check_instance_serialization_and_cloning(concept)
        for concept in document.concepts:
            self.check_instance_serialization_and_cloning(concept)

    # Probabilities of the extracted items of the document created by the method
    # below, by the name of the aspect or concept that an item belongs to
    _document_item_probabilities: ClassVar[
        dict[str, dict[bool | int | str, float] | None]
    ] = {
        # Aspect: whether the paragraph belongs to the aspect
        "Notices": {True: 0.91, False: 0.09},
        # Concept of the aspect: one label out of several
        "Notice form": {
            "written": 0.71,
            "email": 0.20,
            "any": 0.06,
            "not specified": 0.03,
        },
        # Concepts of the document: yes/no question, several labels, rating
        "NDA check": {True: 0.83, False: 0.17},
        "Document types": {"NDA": 0.97, "Confidential": 0.62, "Employment": 0.08},
        "Clarity": {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.55, 5: 0.30},
        # Concept of the document with an item that has no probabilities
        "Title": None,
    }

    def _get_document_with_extracted_items(self, with_probabilities: bool) -> Document:
        """
        Returns a copy of the test document with an aspect and concepts that have
        extracted items assigned, as after the extraction.

        :param with_probabilities: Whether the items have the probabilities of
            the options. If False, no item has them, as when extracted by an LLM.
        :type with_probabilities: bool
        :return: The document with the extracted items.
        :rtype: Document
        """
        probabilities: dict[str, dict[bool | int | str, float] | None] = (
            self._document_item_probabilities
            if with_probabilities
            else dict.fromkeys(self._document_item_probabilities)
        )

        document = self.document.clone()
        document.aspects = [
            Aspect(
                name="Notices",
                description="Clauses on the notices under the agreement",
                concepts=[
                    LabelConcept(
                        name="Notice form",
                        description="Form in which a notice is to be given",
                        labels=["written", "email", "any", "not specified"],
                        classification_type="multi_class",
                    )
                ],
            )
        ]
        document.concepts = [
            BooleanConcept(name="NDA check", description="Document is an NDA"),
            LabelConcept(
                name="Document types",
                description="Types of the document",
                labels=["NDA", "Confidential", "Employment"],
                classification_type="multi_label",
            ),
            RatingConcept(
                name="Clarity",
                description="Clarity of the text of the document",
                rating_scale=(1, 5),
            ),
            StringConcept(name="Title", description="Title of the document"),
        ]

        # Item of the aspect: a paragraph of the document
        aspect = document.get_aspect_by_name("Notices")
        paragraph = document.paragraphs[0]
        aspect_item = _StringItem(
            value=paragraph.raw_text, probabilities=probabilities["Notices"]
        )
        aspect_item.reference_paragraphs = [paragraph]
        aspect.extracted_items = [aspect_item]
        aspect.reference_paragraphs = [paragraph]

        # Items of the concepts
        aspect.get_concept_by_name("Notice form").extracted_items = [
            _LabelItem(value=["written"], probabilities=probabilities["Notice form"])
        ]
        document.get_concept_by_name("NDA check").extracted_items = [
            _BooleanItem(value=True, probabilities=probabilities["NDA check"])
        ]
        document.get_concept_by_name("Document types").extracted_items = [
            _LabelItem(
                value=["NDA", "Confidential"],
                probabilities=probabilities["Document types"],
            )
        ]
        document.get_concept_by_name("Clarity").extracted_items = [
            _IntegerItem(value=4, probabilities=probabilities["Clarity"])
        ]
        document.get_concept_by_name("Title").extracted_items = [
            _StringItem(value="Non-disclosure agreement", probabilities=None)
        ]

        return document

    @staticmethod
    def _get_extracted_items(document: Document) -> dict[str, _ExtractedItem]:
        """
        Returns the extracted items of the document created by
        ``_get_document_with_extracted_items()``, i.e. the single item of each of
        its aspects and concepts.

        :param document: The document with the extracted items.
        :type document: Document
        :return: The extracted items by the name of the aspect or concept that
            an item belongs to.
        :rtype: dict[str, _ExtractedItem]
        """
        instances = [*document.aspects, *document.concepts]
        for aspect in document.aspects:
            instances.extend(aspect.concepts)
        items: dict[str, _ExtractedItem] = {}
        for instance in instances:
            assert len(instance.extracted_items) == 1
            items[instance.name] = instance.extracted_items[0]
        return items

    @classmethod
    def _remove_probabilities_keys(cls, serialized_data: Any) -> int:
        """
        Removes the `probabilities` key from the serialized data of each extracted
        item, in place, which gives the data as it was serialized before
        the probabilities were added to the items.

        :param serialized_data: The serialized data of an instance, or a part of it.
        :type serialized_data: Any
        :return: The number of the keys removed.
        :rtype: int
        """
        removed_keys_count = 0
        if isinstance(serialized_data, dict):
            if "probabilities" in serialized_data:
                assert serialized_data["__class__"].endswith("Item")
                del serialized_data["probabilities"]
                removed_keys_count += 1
            for value in serialized_data.values():
                removed_keys_count += cls._remove_probabilities_keys(value)
        elif isinstance(serialized_data, list):
            for value in serialized_data:
                removed_keys_count += cls._remove_probabilities_keys(value)
        return removed_keys_count

    @pytest.mark.parametrize("restore_method", ["from_dict", "from_json", "from_disk"])
    def test_restored_document_references_its_paragraphs_and_sentences(
        self, restore_method: str, tmp_path: Path
    ):
        """
        Tests that the reference paragraphs and reference sentences of the aspects
        of a document restored from its serialized form, of their sub-aspects,
        and of the extracted items of the aspects and of the concepts at both
        levels, are the paragraphs and sentences of the restored document itself,
        i.e. the same objects, with each of ``from_dict()``, ``from_json()`` and
        ``from_disk()``: when the document is restored, and after more is
        extracted from the restored document. The restored document is equal to
        the one that was serialized, and is serialized to the same data.

        The serialized data has each reference on its own, and each was restored
        as a separate object: an equal copy of a paragraph or of a sentence of
        the document, with the same unique ID.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        # The sentences of each paragraph of the document
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price.", "Payment is due in 14 days."],
            ["Either party may terminate.", "The notice period is 60 days."],
            ["This agreement is governed by the laws of Norway."],
        ]

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API. An aspect gets two of the paragraphs that
            the prompt lists, starting with the one at the position of the aspect
            in the prompt, each with its first sentence if the prompt lists
            sentences. A concept gets one item, which references the first
            paragraph that the prompt lists, and its first sentence if
            the prompt lists sentences.
            """
            prompt = kwargs["messages"][-1]["content"]
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            aspect_ids = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            n_paragraphs = len(re.findall(r"<paragraph_id>P\d+</paragraph_id>", prompt))
            assert n_paragraphs >= 2
            with_sentences = bool(
                re.search(r"<sentence_id>P\d+-S\d+</sentence_id>", prompt)
            )
            content: list[dict[str, Any]] = []
            for concept_id in concept_ids:
                references: dict[str, Any] = (
                    {
                        "reference_paragraphs": [
                            {
                                "reference_paragraph_id": "P0",
                                "reference_sentence_ids": ["P0-S0"],
                            }
                        ]
                    }
                    if with_sentences
                    else {"reference_paragraph_ids": ["P0"]}
                )
                content.append(
                    {
                        "concept_id": concept_id,
                        "extracted_items": [{"value": "30 days", **references}],
                    }
                )
            for aspect_position, aspect_id in enumerate(aspect_ids):
                paragraph_ids = sorted(
                    f"P{(aspect_position + shift) % n_paragraphs}" for shift in range(2)
                )
                content.append(
                    {
                        "aspect_id": aspect_id,
                        "paragraphs": [
                            {
                                "paragraph_id": paragraph_id,
                                "sentences": [{"sentence_id": f"{paragraph_id}-S0"}],
                            }
                            for paragraph_id in paragraph_ids
                        ],
                    }
                    if with_sentences
                    else {"aspect_id": aspect_id, "paragraph_ids": paragraph_ids}
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_concept(
            name: str, reference_depth: Literal["paragraphs", "sentences"]
        ) -> StringConcept:
            """
            Creates a concept with references of the given depth.
            """
            return StringConcept(
                name=name,
                description=f"Description of {name}",
                add_references=True,
                reference_depth=reference_depth,
            )

        def create_aspect(
            name: str, reference_depth: Literal["paragraphs", "sentences"]
        ) -> Aspect:
            """
            Creates an aspect with a sub-aspect, each with a concept.
            """
            return Aspect(
                name=name,
                description=f"Description of {name}",
                reference_depth=reference_depth,
                concepts=[create_concept(f"Concept of {name}", reference_depth)],
                aspects=[
                    Aspect(
                        name=f"Sub-aspect of {name}",
                        description=f"Description of the sub-aspect of {name}",
                        reference_depth=reference_depth,
                        concepts=[
                            create_concept(
                                f"Concept of the sub-aspect of {name}", reference_depth
                            )
                        ],
                    )
                ],
            )

        def check_references(document: Document) -> tuple[int, int, int]:
            """
            Checks that each reference paragraph and each reference sentence of
            the aspects of the document, at both levels, and of the extracted
            items of the aspects and of the concepts, is a paragraph or
            a sentence of the document itself, i.e. the same object. Returns
            the numbers of the aspects and items checked, of their reference
            paragraphs, and of their reference sentences.
            """
            holders: list[Any] = []
            concepts = list(document.concepts)
            for aspect in document.aspects:
                for instance in (aspect, *aspect.aspects):
                    assert instance._is_processed
                    holders.extend([instance, *instance.extracted_items])
                    concepts.extend(instance.concepts)
            for concept in concepts:
                holders.extend(concept.extracted_items)
            n_paragraphs = n_sentences = 0
            for holder in holders:
                for paragraph in holder.reference_paragraphs:
                    position = document.get_paragraph_index(paragraph)
                    assert paragraph is document.paragraphs[position]
                    n_paragraphs += 1
                for sentence in holder.reference_sentences:
                    paragraph_position, position = document.get_sentence_index(sentence)
                    document_paragraph = document.paragraphs[paragraph_position]
                    assert sentence is document_paragraph.sentences[position]
                    n_sentences += 1
            return len(holders), n_paragraphs, n_sentences

        def restore(document: Document, file_name: str) -> Document:
            """
            Serializes the document and restores it with the method under test.
            Checks that the restored document is equal to the document, and is
            serialized to the same data.
            """
            serialized_dict = document.to_dict()
            serialized_json = document.to_json()
            disk_path = tmp_path / file_name
            document.to_disk(disk_path)
            if restore_method == "from_dict":
                restored_document = Document.from_dict(serialized_dict)
            elif restore_method == "from_json":
                restored_document = Document.from_json(serialized_json)
            else:
                restored_document = Document.from_disk(disk_path)
            assert restored_document is not document
            assert restored_document == document
            assert restored_document.to_dict() == serialized_dict
            assert restored_document.to_json() == serialized_json
            restored_disk_path = tmp_path / f"restored_{file_name}"
            restored_document.to_disk(restored_disk_path)
            assert restored_disk_path.read_bytes() == disk_path.read_bytes()
            # The paragraphs of the restored document are its own
            assert not any(
                paragraph is document_paragraph
                for paragraph, document_paragraph in zip(
                    restored_document.paragraphs, document.paragraphs, strict=True
                )
            )
            return restored_document

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            for reference_depth in ("paragraphs", "sentences"):
                # With sentences as the references, each item of an aspect and of
                # a concept references a sentence, and each aspect the sentences
                # of its two items
                with_sentences = reference_depth == "sentences"
                llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
                document = Document(
                    paragraphs=[
                        Paragraph(
                            raw_text=" ".join(texts),
                            sentences=[Sentence(raw_text=text) for text in texts],
                        )
                        for texts in sentence_texts
                    ]
                )
                document.aspects = [
                    create_aspect(f"Aspect {i}", reference_depth) for i in range(2)
                ]
                document.concepts = [
                    create_concept("Document concept", reference_depth)
                ]
                llm.extract_all(document)

                # === The restored document ===
                restored_document = restore(document, f"{reference_depth}.json")
                # 2 aspects and 2 sub-aspects, each with 2 items, and 5 concepts,
                # each with an item
                all_extracted = (17, 21, 21 if with_sentences else 0)
                assert check_references(restored_document) == all_extracted

                # === More is extracted from the restored document ===
                # An aspect with a sub-aspect, and a concept at each level
                restored_document.add_aspects(
                    [create_aspect("Added aspect", reference_depth)]
                )
                restored_document.add_concepts(
                    [create_concept("Added document concept", reference_depth)]
                )
                added_aspect = restored_document.get_aspect_by_name("Added aspect")
                llm.extract_aspects_from_document(
                    restored_document, from_aspects=[added_aspect]
                )
                llm.extract_concepts_from_aspect(added_aspect, restored_document)
                llm.extract_concepts_from_document(
                    restored_document,
                    from_concepts=[
                        restored_document.get_concept_by_name("Added document concept")
                    ],
                )
                # 3 aspects and 3 sub-aspects, each with 2 items, and 8 concepts,
                # each with an item
                more_extracted = (26, 32, 32 if with_sentences else 0)
                assert check_references(restored_document) == more_extracted

                # ... and the document is restored again
                restored_document = restore(
                    restored_document, f"{reference_depth}_more.json"
                )
                assert check_references(restored_document) == more_extracted
                # What was extracted first is restored as it was extracted
                for restored_aspect, aspect in zip(
                    restored_document.aspects[:2], document.aspects, strict=True
                ):
                    assert restored_aspect == aspect

    def test_restored_document_references_in_inconsistent_serialized_data(self):
        """
        Tests how a document is restored from serialized data in which
        a reference is not an equal copy of a paragraph of the document:

        - a reference with the unique ID of a paragraph of the document and with
          other content, e.g. without the sentences that the paragraph has, is
          restored as the paragraph of the document, i.e. the document is
          serialized again with the content of its paragraph in the reference;
        - a reference with a unique ID that no paragraph or sentence of
          the document has is restored as a separate object, as it is serialized.

        Data of the first kind was serialized when the references were copies:
        e.g. when the sentences of a document were segmented after a sub-aspect
        was extracted, the paragraphs of the sub-aspect were left without them.
        """
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price.", "Payment is due in 14 days."],
        ]
        document = Document(
            paragraphs=[
                Paragraph(
                    raw_text=" ".join(texts),
                    sentences=[Sentence(raw_text=text) for text in texts],
                )
                for texts in sentence_texts
            ]
        )
        document.aspects = [Aspect(name="Duties", description="Duties of the parties")]
        (aspect,) = document.aspects
        items = []
        for paragraph in document.paragraphs:
            item = _StringItem(value=paragraph.raw_text)
            item.reference_paragraphs = [paragraph]
            items.append(item)
        aspect.extracted_items = items  # ty: ignore[invalid-assignment]
        aspect.reference_paragraphs = list(document.paragraphs)
        aspect._is_processed = True
        serialized = document.to_dict()
        assert Document.from_dict(serialized).to_dict() == serialized

        # === A reference with other content than the paragraph of the document ===
        stale_serialized = deepcopy(serialized)
        aspect_dict = stale_serialized["aspects"][0]
        item_dict = aspect_dict["_extracted_items"][0]
        for paragraph_dict in (
            aspect_dict["_reference_paragraphs"][0],
            item_dict["_reference_paragraphs"][0],
        ):
            assert paragraph_dict == serialized["paragraphs"][0]
            assert len(paragraph_dict["sentences"]) == 2
            paragraph_dict["sentences"] = []
        assert stale_serialized != serialized
        restored_document = Document.from_dict(stale_serialized)
        (restored_aspect,) = restored_document.aspects
        for position, paragraph in enumerate(restored_document.paragraphs):
            assert restored_aspect.reference_paragraphs[position] is paragraph
            (item_paragraph,) = restored_aspect.extracted_items[
                position
            ].reference_paragraphs
            assert item_paragraph is paragraph
            assert len(paragraph.sentences) == len(sentence_texts[position])
        # The document is equal to the one that has no such copies
        assert restored_document == document
        assert restored_document.to_dict() == serialized

        # === A reference that the document has no paragraph or sentence for ===
        other_paragraph = Paragraph(
            raw_text="Another text. Of another document.",
            sentences=[
                Sentence(raw_text="Another text."),
                Sentence(raw_text="Of another document."),
            ],
        )
        other_serialized = deepcopy(serialized)
        aspect_dict = other_serialized["aspects"][0]
        aspect_dict["_reference_paragraphs"].insert(1, other_paragraph.to_dict())
        aspect_dict["_reference_sentences"] = [other_paragraph.sentences[1].to_dict()]
        restored_document = Document.from_dict(other_serialized)
        (restored_aspect,) = restored_document.aspects
        first_paragraph, restored_other_paragraph, second_paragraph = (
            restored_aspect.reference_paragraphs
        )
        assert first_paragraph is restored_document.paragraphs[0]
        assert second_paragraph is restored_document.paragraphs[1]
        assert restored_other_paragraph == other_paragraph
        assert restored_other_paragraph is not other_paragraph
        assert restored_aspect.reference_sentences == [other_paragraph.sentences[1]]
        assert restored_document.to_dict() == other_serialized
