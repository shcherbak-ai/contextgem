# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for aspects and concepts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Literal, Optional  # noqa: UP035

import pytest
from pydantic import BaseModel, Field, field_validator

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
from contextgem.internal.loggers import logger
from contextgem.internal.typings.types import ItemProbabilities
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    JsonObjectClassStruct,
    JsonObjectConcept,
    JsonObjectExample,
    LabelConcept,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    StringConcept,
    StringExample,
)
from tests.base import BaseTest
from tests.utils import get_project_root_path, get_test_document_text


class TestAspectsAndConcepts(BaseTest):
    """
    Tests for the initialization and validation of aspects, concepts, examples
    and extracted items, and for the extraction of concepts of specific types.
    """

    # Shared objects passed to the `parametrize` decorators of the tests below.
    # The decorators are evaluated in the class body, where the attributes inherited
    # from the base class are not in scope, so the names are bound here.
    llm_extractor = BaseTest.llm_extractor
    model_group = BaseTest.model_group

    def test_init_aspect(self):
        """
        Tests the initializing and error handling of the `Aspect` class.
        """

        with pytest.raises(ValueError):
            Aspect()
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                model_role="",  # invalid model role (empty label)
            )
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                modality="image",  # ty: ignore[unknown-argument]  # aspects are extracted from text only
            )
        aspect = Aspect(
            name="Categories of confidential information",
            description="Clauses describing confidential information covered by the NDA",
            add_justifications=True,
        )
        aspect_concepts = [
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
            ),
            StringConcept(
                name="Technical Information",
                description="Categories of Technical Information",
                model_role="reasoner",
            ),
        ]
        aspect.concepts = aspect_concepts
        assert aspect.concepts is not aspect_concepts
        # Validate assignment
        with pytest.raises(ValueError, match="non-empty list"):
            aspect.add_aspects([])  # empty list
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [  # ty: ignore[invalid-argument-type]
                    1,  # invalid type
                ]
            )
        with pytest.raises(ValueError, match="non-empty list"):
            aspect.add_concepts([])  # empty list
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [
                    StringConcept(  # duplicate name and description
                        name="Business Information",
                        description="Categories of Business Information",
                        add_justifications=True,
                    )
                ]
            )
        with pytest.raises(ValueError, match="extracted from text only"):
            aspect.add_concepts(
                [
                    StringConcept(
                        name="Random",
                        description="Random",
                        modality="image",  # invalid modality for aspect-level concept
                    )
                ]
            )
        with pytest.raises(ValueError, match="extracted from text only"):
            aspect.add_concepts(
                [
                    StringConcept(
                        name="Random",
                        description="Random",
                        modality="multimodal",  # invalid modality for aspect-level concept
                    )
                ]
            )

        # Invalid sequence types
        with pytest.raises(ValueError, match="list"):
            aspect.add_concepts(
                (
                    StringConcept(
                        name="Random",
                        description="Random",
                    ),
                )  # tuple instead of list  # type: ignore
            )
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = (
                StringConcept(
                    name="Random",
                    description="Random",
                ),
            )  # tuple instead of list
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = set([1, 2, 3])  # set instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = range(10)  # range instead of list  # type: ignore

        # Test adding a sub-aspect with the same name or description as the main aspect
        sub_aspect_same_name = Aspect(
            name="Categories of confidential information",  # Same name as parent aspect
            description="Different description",
        )
        with pytest.raises(ValueError):
            aspect.add_aspects([sub_aspect_same_name])

        sub_aspect_same_description = Aspect(
            name="Different name",
            description="Clauses describing confidential information covered by the NDA",  # Same description as parent aspect
        )
        with pytest.raises(ValueError):
            aspect.add_aspects([sub_aspect_same_description])

        # Test that only one level of nesting is allowed for sub-aspects
        main_aspect = Aspect(
            name="Main Aspect",
            description="Main aspect description",
        )
        sub_aspect = Aspect(
            name="Sub Aspect",
            description="Sub aspect description",
            model_role="reasoner",
        )
        sub_sub_aspect = Aspect(
            name="Sub-Sub Aspect",
            description="Sub-sub aspect description",
        )

        # Adding a sub-aspect to the main aspect should work
        main_aspect.add_aspects([sub_aspect])
        assert len(main_aspect.aspects) == 1
        assert main_aspect._nesting_level == 0

        sub_aspect = main_aspect.aspects[0]
        assert sub_aspect._nesting_level == 1

        # Trying to add a sub-sub-aspect to the sub-aspect will raise ValueError
        with pytest.raises(ValueError, match="maximum nesting level"):
            sub_aspect.add_aspects(
                [sub_sub_aspect]
            )  # would be nesting level 2, max is 1

        # Verify no changes occurred due to invalid assignment
        logger.debug(sub_aspect.aspects)  # ty: ignore[invalid-argument-type]
        assert not sub_aspect.aspects

        # Verify no changes occurred due to invalid assignments
        assert not aspect.aspects

        # Invalid extracted items
        with pytest.raises(ValueError):
            aspect.extracted_items = [1, True]  # type: ignore
        # No changes due to invalid assignments
        assert aspect.concepts == aspect_concepts
        assert aspect.concepts is not aspect_concepts
        with pytest.raises(ValueError):
            aspect.get_concept_by_name("Non-existent")  # not assigned concept

        # Adding concepts
        aspect.add_concepts(
            [
                StringConcept(
                    name="Personnel Information",
                    description="Categories of information on personnel and staff",
                    add_justifications=True,
                )
            ]
        )
        assert len(aspect.concepts) == 3
        assert aspect.get_concepts_by_names(
            ["Business Information", "Technical Information", "Personnel Information"]
        )

        # Removal of concepts
        aspect.remove_concept_by_name("Technical Information")
        assert len(aspect.concepts) == 2
        assert len(aspect_concepts) == 2
        with pytest.raises(ValueError):
            aspect.get_concept_by_name("Technical Information")  # not assigned concept
        aspect.remove_all_concepts()
        assert len(aspect.concepts) == 0
        assert len(aspect_concepts) == 2

        # List field items' unique IDs
        concept = aspect_concepts[0]
        with pytest.raises(ValueError):
            aspect.add_concepts([concept, concept])

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(aspect)

    def test_init_and_validate_string_concept(self):
        """
        Tests the initialization of the StringConcept class with valid and invalid input parameters.
        """
        # Valid initialization
        string_concept = StringConcept(
            name="Business Information",
            description="Categories of Business Information",
            add_justifications=True,
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            StringConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                extra_param=True,  # type: ignore
            )

        # Test attaching extracted items
        string_concept.extracted_items = [_StringItem(value="Confidential data")]  # ty: ignore[invalid-assignment]

        # Test with invalid items
        with pytest.raises(ValueError):
            string_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _BooleanItem(value=True)
            ]  # must be _StringItem

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(string_concept)

    def test_init_and_validate_boolean_concept(self):
        """
        Tests the initialization of the BooleanConcept class with valid and invalid input parameters.
        """
        # Valid initialization
        boolean_concept = BooleanConcept(
            name="Contract type check",
            description="Contract is an NDA",
            add_justifications=True,
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            BooleanConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                extra_param=True,  # type: ignore
            )

        # Test attaching extracted items
        boolean_concept.extracted_items = [_BooleanItem(value=True)]  # ty: ignore[invalid-assignment]

        # Test with invalid items
        with pytest.raises(ValueError):
            boolean_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _StringItem(value="True")
            ]  # must be _BooleanItem
        with pytest.raises(ValueError):
            boolean_concept.extracted_items = [  # type: ignore
                _BooleanItem(value=None)  # type: ignore
            ]  # must be _BooleanItem

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(boolean_concept)

    @pytest.mark.vcr
    def test_extract_bool_values(self):
        """
        Tests the extraction of a BooleanConcept instance.
        """
        concept_positive_1 = BooleanConcept(
            name="NDA check",
            description="Document is an NDA",
            add_justifications=True,
        )
        concept_positive_2 = BooleanConcept(
            name="Liability section check",
            description="Document has a section on liability",
            add_justifications=False,
            add_references=True,
        )
        concept_negative_1 = BooleanConcept(
            name="Supplier agreement check",
            description="Document is a supplier agreement",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )
        concept_negative_2 = BooleanConcept(
            name="Payment terms section check",
            description="Document includes a section on payment terms",
            add_justifications=False,
        )
        concept_impossible_to_determine_1 = BooleanConcept(
            name="Bitcoin price check",
            description="Bitcoin price is above $100,000",
            add_justifications=True,
        )
        concept_impossible_to_determine_2 = BooleanConcept(
            name="Weather check",
            description="Weather in Oslo is above 30 degrees today",
            add_justifications=False,
        )
        document = self.document.clone()
        document.add_concepts(
            [
                concept_positive_1,
                concept_positive_2,
                concept_negative_1,
                concept_negative_2,
                concept_impossible_to_determine_1,
                concept_impossible_to_determine_2,
            ]
        )
        (
            concept_positive_1,
            concept_positive_2,
            concept_negative_1,
            concept_negative_2,
            concept_impossible_to_determine_1,
            concept_impossible_to_determine_2,
        ) = self.llm_extractor.extract_concepts_from_document(
            document, use_concurrency=True
        )
        assert concept_positive_1.extracted_items
        assert concept_positive_2.extracted_items
        assert concept_negative_1.extracted_items
        assert concept_negative_2.extracted_items
        assert concept_positive_1.extracted_items[0].value is True
        assert concept_positive_2.extracted_items[0].value is True
        assert concept_negative_1.extracted_items[0].value is False
        assert concept_negative_2.extracted_items[0].value is False
        assert (
            not concept_impossible_to_determine_1.extracted_items
            or concept_impossible_to_determine_1.extracted_items[0].value is False
        )
        assert (
            not concept_impossible_to_determine_2.extracted_items
            or concept_impossible_to_determine_2.extracted_items[0].value is False
        )

        # Test multiple items of the same concept
        text = """
        CONSOLIDATED AUDIT REPORT - Q3 2024

        SUBSIDIARY COMPLIANCE REVIEW:

        1. TechCorp North America: The audit identified several GDPR compliance gaps 
        in data processing procedures. Immediate remediation required.

        2. XYZ Europe Ltd: All regulatory requirements met. No compliance issues 
        identified during the review period.

        3. ABC Asia Pacific: GDPR compliance satisfactory, however, local data 
        residency requirements in Singapore market not fully addressed.
        """
        document = Document(raw_text=text)
        concept_multiple_items = BooleanConcept(
            name="Has Regulatory Compliance Issues",
            description="Entity has outstanding regulatory compliance violations or issues. "
            "Analyze and extract the value for each mentioned entity.",
            singular_occurrence=False,  # Allow multiple items (default)
            add_references=True,
            reference_depth="sentences",
        )
        document.add_concepts([concept_multiple_items])
        (concept_multiple_items,) = self.llm_extractor.extract_concepts_from_document(
            document
        )
        assert len(concept_multiple_items.extracted_items) == 3
        assert concept_multiple_items.extracted_items[0].value is True
        assert concept_multiple_items.extracted_items[1].value is False
        assert concept_multiple_items.extracted_items[2].value is True

        # Check that we have identical behaviour when bool is a return type
        # in a JsonObjectConcept field
        json_obj_concept_positive = JsonObjectConcept(
            name="NDA check",
            description="Document is an NDA",
            add_justifications=True,
            structure={
                "is_nda": bool,
            },
        )
        json_obj_concept_negative = JsonObjectConcept(
            name="Supplier agreement check",
            description="Document is a supplier agreement",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
            structure={
                "is_supplier_agreement": bool,
            },
        )
        json_obj_concept_impossible_to_determine = JsonObjectConcept(
            name="Bitcoin price check",
            description="Bitcoin price is above $100,000",
            add_justifications=True,
            structure={
                "bitcoin_price_is_above_100k": bool,
            },
        )
        document = self.document.clone()
        document.add_concepts(
            [
                json_obj_concept_positive,
                json_obj_concept_negative,
                json_obj_concept_impossible_to_determine,
            ]
        )
        (
            json_obj_concept_positive,
            json_obj_concept_negative,
            json_obj_concept_impossible_to_determine,
        ) = self.llm_extractor.extract_concepts_from_document(
            document, use_concurrency=True
        )
        assert json_obj_concept_positive.extracted_items
        assert json_obj_concept_negative.extracted_items
        assert json_obj_concept_positive.extracted_items[0].value["is_nda"] is True
        assert (
            json_obj_concept_negative.extracted_items[0].value["is_supplier_agreement"]
            is False
        )
        assert (
            not json_obj_concept_impossible_to_determine.extracted_items
            or json_obj_concept_impossible_to_determine.extracted_items[0].value[
                "bitcoin_price_is_above_100k"
            ]
            is False
        )

    def test_init_and_validate_numerical_concept(self):
        """
        Tests for initialization and usage of NumericalConcept with different types.
        """
        # Test initialization with different numeric_type values
        int_concept = NumericalConcept(
            name="Integer concept",
            description="Test integer concept",
            numeric_type="int",
        )
        float_concept = NumericalConcept(
            name="Float concept", description="Test float concept", numeric_type="float"
        )
        any_concept = NumericalConcept(
            name="Any numeric concept",
            description="Test any numeric concept",
            numeric_type="any",
        )

        # Test with invalid numeric_type
        with pytest.raises(ValueError):
            NumericalConcept(
                name="Invalid type",
                description="Test invalid type",
                numeric_type="string",  # type: ignore
            )

        # Test attaching extracted items
        # Integer concept should accept only integer items
        int_item = _IntegerItem(value=42)
        int_concept.extracted_items = [int_item]  # ty: ignore[invalid-assignment]

        # Should reject float items for int concept
        with pytest.raises(ValueError):
            int_concept.extracted_items = [_FloatItem(value=42.5)]  # ty: ignore[invalid-assignment]

        # Float concept should accept only float items
        float_item = _FloatItem(value=3.14)
        float_concept.extracted_items = [float_item]  # ty: ignore[invalid-assignment]

        # Should reject integer items for float concept
        with pytest.raises(ValueError):
            float_concept.extracted_items = [_IntegerItem(value=3)]  # ty: ignore[invalid-assignment]

        # Any numeric concept should accept both integer and float items
        any_concept.extracted_items = [_IntegerOrFloatItem(value=20)]  # ty: ignore[invalid-assignment]
        any_concept.extracted_items = [_IntegerOrFloatItem(value=20.5)]  # ty: ignore[invalid-assignment]
        # Should reject integer items for any numeric concept
        with pytest.raises(ValueError):
            any_concept.extracted_items = [_IntegerItem(value=10)]  # ty: ignore[invalid-assignment]
        # Should reject float items for any numeric concept
        with pytest.raises(ValueError):
            any_concept.extracted_items = [_FloatItem(value=10.5)]  # ty: ignore[invalid-assignment]

        # Test with justifications and references
        concept_with_refs = NumericalConcept(
            name="Concept with refs",
            description="Test concept with references",
            numeric_type="any",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(int_concept)
        self.check_instance_serialization_and_cloning(float_concept)
        self.check_instance_serialization_and_cloning(any_concept)
        self.check_instance_serialization_and_cloning(concept_with_refs)

    def test_init_and_validate_rating_concept(self):
        """
        Tests the initialization of the RatingConcept class with valid and invalid input parameters.
        """

        # Valid initialization
        valid_rating_concept = RatingConcept(
            name="Customer Satisfaction",
            description="Rating of customer satisfaction",
            rating_scale=(1, 5),
        )

        # Test property access
        assert valid_rating_concept._rating_start == 1
        assert valid_rating_concept._rating_end == 5

        # Invalid initialization
        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(),  # empty tuple  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(-1, 5),  # negative start
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(1, 0),  # zero end
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(5, 5),  # equal start and end
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(10, 5),  # start greater than end
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(
                    1,
                ),  # tuple with invalid length (1 element)  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(
                    1,
                    1,
                    2,
                ),  # tuple with invalid length (3 elements)  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(1.5, 3),  # float element  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale={"start": 1, "end": 5},  # not a tuple  # type: ignore
            )

        # Test attaching extracted items
        rating_item = _IntegerItem(value=3)
        valid_rating_concept.extracted_items = [rating_item]

        # Duplicated items
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [rating_item, rating_item]

        # Test with out-of-range values
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [
                _IntegerItem(value=0)
            ]  # below min (1)

        with pytest.raises(ValueError):
            valid_rating_concept._process_item_value(0)

        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [
                _IntegerItem(value=6)
            ]  # above max (5)

        with pytest.raises(ValueError):
            valid_rating_concept._process_item_value(6)

        # Test with non-integer items
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [_FloatItem(value=3.5)]  # type: ignore

        # Test with justifications and references
        concept_with_refs = RatingConcept(
            name="Concept with refs",
            description="Test concept with references",
            rating_scale=(1, 10),
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(valid_rating_concept)
        self.check_instance_serialization_and_cloning(concept_with_refs)

    def test_init_and_validate_json_object_concept(self):
        """
        Tests the initialization of the JsonObjectConcept class with valid and invalid input parameters.
        """
        # Base class direct initialization
        with pytest.raises(TypeError):
            _Concept(name="Test", description="Test")
        with pytest.raises(ValueError):
            JsonObjectConcept()
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "test": str,
                },
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "_test": str,  # cannot start with underscores as these attrs will not be validated by pydantic
                    "info": str,
                },
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "test": str,
                },
                modality="image",
                add_references=True,
                # references are not supported in vision concepts
            )

        # Define all structures to test
        # 1. Simple dictionary structure
        dict_struct = {
            "category": str,
            "descriptions": List[str],  # noqa: UP006
        }

        # 2. Class structure
        class Struct:
            category: str
            descriptions: list[str]

        # 3. Pydantic structure
        class PydanticStruct(BaseModel):
            category: str
            descriptions: list[str]

        # 4. Structure with optional types
        optional_struct = {
            "required_field": str,
            "optional_str": Optional[str],  # noqa: UP045
            "optional_int": Optional[int],  # noqa: UP045
            "optional_float": float | None,
            "optional_bool": bool | None,
            "union_syntax": str | None,
        }

        # 5. Structure with literal types
        literal_struct = {
            "status": Literal["active", "inactive", "pending"],
            "role": Literal["admin", "user", "guest"],
            "priority": Literal[1, 2, 3],
        }

        # 6. Nested dictionary structure
        nested_dict_struct = {
            "name": str,
            "age": int,
            "contact": {
                "email": str,
                "phone": str,
                "address": {"street": str, "city": str, "country": str},
            },
        }

        # 7. Structure with list of dictionaries
        list_of_dicts_struct = {
            "name": str,
            "skills": [{"name": str, "level": int}],
        }

        # 8. Classes with type annotations for nested structures
        # We also add fields to test warnings (field logic will be discarded)
        @dataclass
        class Address(JsonObjectClassStruct):
            street: str = field(metadata={"description": "Street address"})
            city: str = field(repr=True)
            country: str = field(compare=True)

        @dataclass
        class Contact(JsonObjectClassStruct):
            email: str = field(metadata={"description": "test"})
            phone: str = field(metadata={"format": "international"})
            address: Address = field(repr=False)
            contact_type: Optional[  # noqa: UP045
                Literal["primary", "secondary", "emergency", None]
                | Literal["union", "literal", None]
            ] = field(default=None)  # intentionally messed up type hint for testing

        @dataclass
        class Person(JsonObjectClassStruct):
            name: str = field(compare=True)
            age: int = field(metadata={"min": 0, "max": 120})
            contact: Contact = field(repr=False)

        # 9. Pydantic models for nested structures
        class AddressModel(BaseModel, JsonObjectClassStruct):
            street: str
            city: str
            country: str

        class ContactModel(BaseModel, JsonObjectClassStruct):
            email: str
            phone: str
            address: AddressModel

        class PersonModel(BaseModel, JsonObjectClassStruct):
            name: str
            age: int
            contact: ContactModel

        # 10. Super complex structure combining everything
        super_complex_struct = {
            "user": PersonModel,
            "status": Literal["active", "inactive"],
            "permissions": [
                {"resource": str, "access": Literal["read", "write", "admin"]}
            ],
            "settings": {
                "theme": str,
                "notifications": {
                    "email": bool,
                    "sms": bool,
                    "frequency": Literal["daily", "weekly", "monthly"],
                },
            },
        }

        # 11. Structure with list[cls]
        @dataclass
        class Item(JsonObjectClassStruct):
            id: int
            name: str
            active: bool

        list_of_class_struct = {"title": str, "items": list[Item]}

        # Class structure containing a list[cls] type hint
        @dataclass
        class ClassWithListOfClasses(JsonObjectClassStruct):
            name: str
            description: str
            items: list[Item]

        # 12. Structure with Dict mapping
        dict_mapping_struct = {
            "name": str,
            "scores": Dict[str, int],  # noqa: UP006
        }

        # 13. Pydantic model with Field() validations
        # We add fields to test warnings (field logic will be discarded)
        class ValidatedModel(BaseModel, JsonObjectClassStruct):
            name: str = Field(..., min_length=3, max_length=50)
            age: int = Field(..., ge=18, le=100)
            email: str = Field(
                ..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            )
            tags: list[str] = Field(default_factory=list, max_length=5)
            score: float = Field(default=0.0, ge=0.0, le=10.0)

        validated_model_struct = ValidatedModel

        # 14. Nested structure with dict having union type values
        @dataclass
        class ConfigSettings(JsonObjectClassStruct):
            enabled: bool
            values: dict[
                str, int | float | str | bool | None
            ]  # Union type as dict value
            description: str

        @dataclass
        class AppConfiguration(JsonObjectClassStruct):
            app_name: str
            version: str
            settings: ConfigSettings

        # 15. Pydantic model with field validators
        class UserProfile(BaseModel, JsonObjectClassStruct):
            name: str
            email: str
            age: int

            @field_validator("email")
            @classmethod
            def validate_email(cls, v):
                if "@" not in v:
                    raise ValueError("Email must contain @ symbol")
                return v

            @field_validator("age")
            @classmethod
            def validate_age(cls, v):
                if v < 0 or v > 120:
                    raise ValueError("Age must be between 0 and 120")
                return v

        # 16. Structure with list instances using primitive types
        list_instance_primitives_struct = {
            "product_names": [Literal["laptop", "mouse", "keyboard"]],
            "scores": [int],
            "prices": [float],
            "active_flags": [bool],
            "mixed_numbers": [int | float],
            "optional_tags": [str | None],
        }

        # 17. Structure with list instances using classes
        @dataclass
        class Product(JsonObjectClassStruct):
            id: int
            name: str
            price: float

        list_instance_classes_struct = {
            "title": str,
            "products": [Product],
            "categories": [Literal["electronics", "books", "clothing"]],
        }

        # Define valid examples for each structure
        valid_examples = {
            "ClassWithListOfClasses": {
                "name": "Collection of Items",
                "description": "A collection of multiple items",
                "items": [
                    {"id": 1, "name": "First Item", "active": True},
                    {"id": 2, "name": "Second Item", "active": False},
                    {"id": 3, "name": "Third Item", "active": True},
                ],
            },
            "dict_struct": {
                "category": "test",
                "descriptions": ["test1", "test2", "test3"],
            },
            "Struct": {
                "category": "test",
                "descriptions": ["test1", "test2", "test3"],
            },
            "PydanticStruct": {
                "category": "test",
                "descriptions": ["test1", "test2", "test3"],
            },
            "optional_struct": {
                "required_field": "value",
                "optional_str": "string value",
                "optional_int": 123,
                "optional_float": 45.67,
                "optional_bool": True,
                "union_syntax": None,
            },
            "literal_struct": {
                "status": "active",
                "role": "admin",
                "priority": 1,
            },
            "nested_dict_struct": {
                "name": "John Doe",
                "age": 30,
                "contact": {
                    "email": "john@example.com",
                    "phone": "123-456-7890",
                    "address": {
                        "street": "123 Main St",
                        "city": "Anytown",
                        "country": "USA",
                    },
                },
            },
            "list_of_dicts_struct": {
                "name": "Jane Smith",
                "skills": [
                    {"name": "Python", "level": 5},
                    {"name": "JavaScript", "level": 4},
                ],
            },
            "Person": {
                "name": "Alice Johnson",
                "age": 28,
                "contact": {
                    "email": "alice@example.com",
                    "phone": "987-654-3210",
                    "address": {
                        "street": "456 Oak Ave",
                        "city": "Somewhere",
                        "country": "Canada",
                    },
                    "contact_type": "primary",
                },
            },
            "PersonModel": {
                "name": "Bob Brown",
                "age": 35,
                "contact": {
                    "email": "bob@example.com",
                    "phone": "555-123-4567",
                    "address": {
                        "street": "789 Pine St",
                        "city": "Elsewhere",
                        "country": "UK",
                    },
                },
            },
            "super_complex_struct": {
                "user": {
                    "name": "Charlie Davis",
                    "age": 42,
                    "contact": {
                        "email": "charlie@example.com",
                        "phone": "111-222-3333",
                        "address": {
                            "street": "101 Maple Dr",
                            "city": "Nowhere",
                            "country": "Australia",
                        },
                    },
                },
                "status": "active",
                "permissions": [
                    {"resource": "files", "access": "read"},
                    {"resource": "users", "access": "admin"},
                ],
                "settings": {
                    "theme": "dark",
                    "notifications": {
                        "email": True,
                        "sms": False,
                        "frequency": "weekly",
                    },
                },
            },
            "list_of_class_struct": {
                "title": "My Item Collection",
                "items": [
                    {"id": 1, "name": "First Item", "active": True},
                    {"id": 2, "name": "Second Item", "active": False},
                ],
            },
            "dict_mapping_struct": {
                "name": "Student Results",
                "scores": {"math": 95, "science": 87, "history": 78},
            },
            "ValidatedModel": {
                "name": "John Doe",
                "age": 35,
                "email": "john.doe@example.com",
                "tags": ["developer", "python"],
                "score": 8.5,
            },
            "AppConfiguration": {
                "app_name": "TestApp",
                "version": "1.0.0",
                "settings": {
                    "enabled": True,
                    "values": {
                        "timeout": 30,
                        "rate_limit": 5.5,
                        "api_key": "abc123",
                        "debug_mode": False,
                        "cache": None,
                    },
                    "description": "Test configuration settings",
                },
            },
            "UserProfile": {
                "name": "John Smith",
                "email": "john@example.com",
                "age": 30,
            },
            "list_instance_primitives_struct": {
                "product_names": ["laptop", "mouse", "keyboard"],
                "scores": [95, 87, 78],
                "prices": [999.99, 25.50, 75.00],
                "active_flags": [True, False, True],
                "mixed_numbers": [10, 15.5, 20],
                "optional_tags": ["sale", None, "premium"],
            },
            "list_instance_classes_struct": {
                "title": "Product Catalog",
                "products": [
                    {"id": 1, "name": "Laptop", "price": 999.99},
                    {"id": 2, "name": "Mouse", "price": 25.50},
                    {"id": 3, "name": "Keyboard", "price": 75.00},
                ],
                "categories": ["electronics", "electronics", "electronics"],
            },
        }

        # Define invalid examples for each structure
        invalid_examples = {
            "ClassWithListOfClasses": [
                {"name": "Test"},  # Missing required fields
                {"name": "Test", "description": "Test"},  # Missing items field
                {"name": "Test", "items": []},  # Missing description field
                {
                    "name": "Test",
                    "description": "Test",
                    "items": {},
                },  # Wrong type for items
                {
                    "name": "Test",
                    "description": "Test",
                    "items": [{"id": "1", "name": "Test", "active": True}],
                },  # Wrong type for id (must be int)
                {
                    "name": "Test",
                    "description": "Test",
                    "items": [{"name": "Test", "active": True}],
                },  # Missing required field in item
            ],
            "dict_struct": [
                {"category": None},  # Missing required field
                {"descriptions": ["test"]},  # Missing required field
                {"category": "test", "descriptions": "not a list"},  # Wrong type
                [],  # Not a dict
            ],
            "Struct": [
                {"category": None},  # Missing required field
                {"descriptions": ["test"]},  # Missing required field
                {"category": "test", "descriptions": "not a list"},  # Wrong type
                [],  # Not a dict
            ],
            "PydanticStruct": [
                {"category": None},  # Missing required field
                {"descriptions": ["test"]},  # Missing required field
                {"category": "test", "descriptions": "not a list"},  # Wrong type
                [],  # Not a dict
            ],
            "optional_struct": [
                {},  # Missing required field
                {"optional_str": "test"},  # Missing required field
                {"required_field": 123},  # Wrong type for required field
                {
                    "required_field": "value",
                    "optional_str": 123,
                },  # Wrong type for optional field
            ],
            "literal_struct": [
                {"status": "unknown"},  # Invalid literal value
                {"role": "superuser"},  # Invalid literal value
                {"priority": 4},  # Invalid literal value
                {
                    "status": "active",
                    "role": "admin",
                    "priority": "1",
                },  # Wrong type for literal
            ],
            "nested_dict_struct": [
                {"name": "John", "contact": {}},  # Missing required field
                {"name": "John", "age": "30"},  # Wrong type
                {
                    "name": "John",
                    "age": 30,
                    "contact": {"email": "john@example.com"},
                },  # Incomplete nested structure
            ],
            "list_of_dicts_struct": [
                {"name": "Jane", "skills": "Python"},  # Wrong type for list
                {
                    "name": "Jane",
                    "skills": [{"name": "Python"}],
                },  # Missing required field in list item
                {
                    "name": "Jane",
                    "skills": [{"level": 5}],
                },  # Missing required field in list item
            ],
            "Person": [
                {"name": "Alice", "age": "28"},  # Wrong type
                {"name": "Alice", "contact": {}},  # Incomplete nested structure
                {
                    "name": "Alice",
                    "age": 28,
                    "contact": {"email": "alice@example.com"},
                },  # Incomplete nested structure
            ],
            "PersonModel": [
                {"name": "Bob", "age": "35"},  # Wrong type
                {"name": "Bob", "contact": {}},  # Incomplete nested structure
                {
                    "name": "Bob",
                    "age": 35,
                    "contact": {"email": "bob@example.com"},
                },  # Incomplete nested structure
            ],
            "super_complex_struct": [
                {"user": {}, "status": "active"},  # Incomplete user structure
                {
                    "user": {"name": "Charlie", "age": 42},
                    "status": "unknown",
                },  # Invalid literal value
                {
                    "user": {"name": "Charlie", "age": 42, "contact": {}},
                    "status": "active",
                },  # Incomplete contact
            ],
            "list_of_class_struct": [
                {"title": "Collection"},  # Missing items field
                {"items": []},  # Missing title field
                {"title": "Collection", "items": {}},  # Wrong type for items
                {
                    "title": "Collection",
                    "items": [{"id": "1", "name": "Item", "active": True}],
                },  # Wrong type for id (must be int)
                {
                    "title": "Collection",
                    "items": [{"name": "Item", "active": True}],
                },  # Missing required field in item
            ],
            "dict_mapping_struct": [
                {"name": "Results"},  # Missing required fields
                {"name": "Results", "scores": []},  # Wrong type for scores
                {
                    "name": "Results",
                    "scores": {"math": "A"},
                },  # Wrong value type in scores
            ],
            "ValidatedModel": [
                {
                    "name": "John Doe",
                    "age": 35,
                    "email": "john.doe@example.com",
                },  # some fields are missing
                {
                    "name": 1,
                    "age": 17,
                    "email": "john.doe@example.com",
                    "tags": ["one", "two", "three", "four", "five", "six"],
                    "score": 11.0,
                },  # name field is of wrong type
            ],
            "AppConfiguration": [
                {
                    "app_name": "TestApp",
                    "version": "1.0.0",
                },  # Incomplete nested structure, missing fields
                {
                    "app_name": "TestApp",
                    "version": "1.0.0",
                    "settings": {
                        "enabled": True,
                        "description": "Test settings",
                    },
                },  # Missing values dict
            ],
            "UserProfile": [
                {"name": "John", "age": 30},  # Missing email
            ],
            "list_instance_primitives_struct": [
                {"product_names": "not a list"},  # Wrong type for list field
                {"scores": [95, "invalid"]},  # Wrong type in list element
                {"prices": [999.99, 25.50]},
                {"active_flags": ["True", False]},  # Wrong type in list element
                {"mixed_numbers": [10, "invalid"]},  # Wrong type in union
                {
                    "product_names": [],
                    "scores": [],
                    "prices": [],
                    "active_flags": [],
                    "mixed_numbers": [],
                    "optional_tags": "not a list",
                },  # Wrong type for optional list
            ],
            "list_instance_classes_struct": [
                {"products": []},  # Missing required fields
                {"title": "Catalog"},  # Missing required fields
                {"title": "Catalog", "products": {}},  # Wrong type for products
                {
                    "title": "Catalog",
                    "products": [{"id": "1", "name": "Test", "price": 10.0}],
                },  # Wrong type for id
                {
                    "title": "Catalog",
                    "products": [{"name": "Test", "price": 10.0}],
                },  # Missing required field in product
                {
                    "title": "Catalog",
                    "products": [],
                    "categories": ["invalid"],
                },  # Invalid literal value
            ],
        }

        # Map structure objects to their names
        structure_map = {
            "dict_struct": dict_struct,
            "Struct": Struct,
            "PydanticStruct": PydanticStruct,
            "optional_struct": optional_struct,
            "literal_struct": literal_struct,
            "nested_dict_struct": nested_dict_struct,
            "list_of_dicts_struct": list_of_dicts_struct,
            "Person": Person,
            "PersonModel": PersonModel,
            "super_complex_struct": super_complex_struct,
            "list_of_class_struct": list_of_class_struct,
            "ClassWithListOfClasses": ClassWithListOfClasses,
            "dict_mapping_struct": dict_mapping_struct,
            "ValidatedModel": validated_model_struct,
            "AppConfiguration": AppConfiguration,
            "UserProfile": UserProfile,
            "list_instance_primitives_struct": list_instance_primitives_struct,
            "list_instance_classes_struct": list_instance_classes_struct,
        }

        # Utility function to test a structure
        def test_structure(struct_name: str, structure: Any) -> JsonObjectConcept:
            """
            Utility function to test a JsonObjectConcept structure.

            :param struct_name: Name of the structure being tested.
            :type struct_name: str
            :param structure: The structure definition to test.
            :type structure: Any
            :return: The JsonObjectConcept instance.
            :rtype: JsonObjectConcept
            """
            # Create concept with the structure
            concept = JsonObjectConcept(
                name=f"{struct_name} Concept",
                description=f"Testing {struct_name} structure",
                structure=structure,
                examples=[JsonObjectExample(content=valid_examples[struct_name])],
            )

            # Test instance serialization and cloning
            self.check_instance_serialization_and_cloning(concept)

            # Test valid extracted data
            validator = concept._get_structure_validator()
            valid_data = validator.model_validate(valid_examples[struct_name])
            assert valid_data is not None

            # Test invalid extracted data
            for invalid_example in invalid_examples[struct_name]:
                with pytest.raises(ValueError):
                    validator.model_validate(invalid_example)

            # Test name assignment
            concept.name = f"Updated {struct_name}"
            with pytest.raises(ValueError):
                concept.name = True  # type: ignore
            assert concept.name == f"Updated {struct_name}"

            # Test invalid extracted items
            with pytest.raises(ValueError):
                concept.extracted_items = [1, True]  # type: ignore

            return concept

        # Test each structure
        for struct_name, structure in structure_map.items():
            test_structure(struct_name, structure)

        # Test invalid structure cases
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={str: str},  # invalid mapping  # type: ignore
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={int: "integer"},  # invalid mapping  # type: ignore
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={},  # empty mapping
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={"random": Exception},  # invalid value type
            )

        # Test with invalid type hints as structure values
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Optional Type",
                description="Test with invalid Optional type",
                structure={"field": Optional[Person]},  # noqa: UP045
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Dict Type",
                description="Test with invalid dict type",
                structure={"field": dict[str, Item]},  # cls as value
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Dict Type",
                description="Test with invalid dict type",
                structure={"field": dict[Item, str]},  # cls as key
            )

        # Test with vision extractor and references (should fail)
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Vision Concept with References",
                description="Testing vision concept with references",
                structure={"test": str},
                modality="image",
                add_references=True,
            )

        # Test with Any type (should fail)
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Any Concept",
                description="Testing Any type",
                structure={"test": Any},  # unclear type
            )

        # Test with raw list and dicts (should fail)
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Raw List and Dicts Concept",
                description="Testing raw list and dicts",
                structure={"test": list},  # unclear type
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Raw List and Dicts Concept",
                description="Testing raw list and dicts",
                structure={"test": dict},  # unclear type
            )

        # Test with unclear types in dataclasses (should fail)
        with pytest.raises(ValueError):

            @dataclass
            class StudentWithUnclearAny(JsonObjectClassStruct):
                name: str
                courses: Any  # unclear type - should fail

            JsonObjectConcept(
                name="Unclear Any Class Concept",
                description="Testing unclear any type in dataclass",
                structure=StudentWithUnclearAny,
            )

        with pytest.raises(ValueError):

            @dataclass
            class StudentWithUnclearDict(JsonObjectClassStruct):
                name: str
                grades: dict  # unclear type - should fail

            JsonObjectConcept(
                name="Unclear Dict Class Concept",
                description="Testing unclear dict type in dataclass",
                structure=StudentWithUnclearDict,
            )

        # Test with vision extractor and justifications (should work)
        vision_concept = JsonObjectConcept(
            name="Vision Concept with Justifications",
            description="Testing vision concept with justifications",
            structure={
                "category": str,
                "description": str,
                "score": int | float | None,
            },
            modality="image",
            add_justifications=True,
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(vision_concept)

    @pytest.mark.vcr
    def test_extract_complex_json_object_concept(self):
        """
        Tests the extraction of a complex JsonObjectConcept from a text file, validating
        that the structure representation is properly included in the prompt and that
        the concept is correctly extracted.
        """

        @dataclass
        class Address(JsonObjectClassStruct):
            street: str
            city: str
            country: str

        @dataclass
        class Contact(JsonObjectClassStruct):
            email: str
            phone: str
            address: Address
            contact_type: (
                Literal["primary", "secondary", "emergency"]
                | Literal["union", "literal"]
                | None
            )  # intentionally messed up type hint for testing

        @dataclass
        class PersonModel(JsonObjectClassStruct):
            name: str
            age: int
            contact: Contact

        # Define an Item class for use in list[cls]
        @dataclass
        class UserItem(JsonObjectClassStruct):
            id: int
            name: str
            description: str
            is_active: bool

        # Define the complex structure
        super_complex_struct = {
            "user": PersonModel,
            "status": Literal["active", "inactive"],
            "permissions": [
                {"resource": str, "access": Literal["read", "write", "admin"]}
            ],
            "settings": {
                "theme": str,
                "notifications": {
                    "email": bool,
                    "sms": bool,
                    "frequency": Literal["daily", "weekly", "monthly"],
                },
            },
            "related_items": [
                UserItem
            ],  # Add [cls] nested structure (equivalent to list[cls])
            "security_level": Literal["Basic", "Advanced", "Enterprise"] | None,
        }

        # Create a JsonObjectConcept with this structure
        complex_concept = JsonObjectConcept(
            name="User System Profile",
            description="Comprehensive user profile with system access permissions and settings",
            structure=super_complex_struct,
            examples=[
                JsonObjectExample(
                    content={
                        "user": {
                            "name": "Charlie Davis",
                            "age": 42,
                            "contact": {
                                "email": "charlie@example.com",
                                "phone": "111-222-3333",
                                "address": {
                                    "street": "101 Maple Dr",
                                    "city": "Nowhere",
                                    "country": "Australia",
                                },
                                "contact_type": "primary",
                            },
                        },
                        "status": "active",
                        "permissions": [
                            {"resource": "files", "access": "read"},
                            {"resource": "users", "access": "admin"},
                        ],
                        "settings": {
                            "theme": "dark",
                            "notifications": {
                                "email": True,
                                "sms": False,
                                "frequency": "weekly",
                            },
                        },
                        "related_items": [
                            {
                                "id": 1,
                                "name": "Document Review",
                                "description": "Annual document review task",
                                "is_active": True,
                            },
                            {
                                "id": 2,
                                "name": "Access Audit",
                                "description": "System access verification",
                                "is_active": False,
                            },
                        ],
                        "security_level": "Advanced",
                    }
                )
            ],
        )

        # Load the test text file as a Document
        with open(
            os.path.join(
                get_project_root_path(),
                "tests",
                "other_files",
                "complex_user_profile.txt",
            ),
            encoding="utf-8",
        ) as f:
            text_content = f.read()

        # Create a Document from the text content and add the complex concept
        document = Document(raw_text=text_content)
        document.concepts = [complex_concept]

        # Get the structure representation that should be included in the prompt
        structure_str = complex_concept._format_structure_in_prompt()

        # Verify that the structure representation contains key elements
        assert '"user"' in structure_str
        assert '"name": str' in structure_str
        assert '"age": int' in structure_str
        assert '"status": "active" or "inactive"' in structure_str
        assert '"permissions"' in structure_str
        assert '"resource": str' in structure_str
        assert '"access": "read" or "write" or "admin"' in structure_str
        assert '"theme": str' in structure_str
        assert '"email": bool' in structure_str
        assert '"frequency": "daily" or "weekly" or "monthly"' in structure_str
        assert (
            '"security_level": "Basic" or "Advanced" or "Enterprise" or null'
            in structure_str
        )
        assert (
            '"contact_type": "primary" or "secondary" or "emergency" or "union" or "literal" or null'
            in structure_str
        )

        # Verify the list[cls] structure representation
        assert '"related_items"' in structure_str
        assert '"id": int' in structure_str
        assert '"description": str' in structure_str
        assert '"is_active": bool' in structure_str

        # Extract the concept
        extracted_concepts = self.llm_extractor.extract_concepts_from_document(document)

        # Verify prompt content from the LLM call log
        prompt_string = self.llm_extractor.get_usage()[-1].usage.calls[-1].prompt
        assert structure_str in prompt_string

        # Validate extraction results
        assert len(extracted_concepts) == 1
        extracted_concept = extracted_concepts[0]
        assert extracted_concept.extracted_items
        extracted_item = extracted_concept.extracted_items[0]
        extracted_data = extracted_item.value

        # Validate the extracted data structure
        assert "user" in extracted_data
        assert extracted_data["user"]["name"] == "Charlie Davis"
        assert extracted_data["user"]["age"] == 42
        assert extracted_data["user"]["contact"]["email"] == "charlie@example.com"
        assert extracted_data["user"]["contact"]["address"]["city"] == "Nowhere"
        assert extracted_data["user"]["contact"]["contact_type"] == "primary"
        assert extracted_data["status"] == "active"
        assert isinstance(extracted_data["permissions"], list)
        assert len(extracted_data["permissions"]) >= 1
        assert "resource" in extracted_data["permissions"][0]
        assert "access" in extracted_data["permissions"][0]
        assert "settings" in extracted_data
        assert "theme" in extracted_data["settings"]
        assert "notifications" in extracted_data["settings"]
        assert "email" in extracted_data["settings"]["notifications"]
        assert "frequency" in extracted_data["settings"]["notifications"]
        assert "security_level" in extracted_data
        assert extracted_data["security_level"] == "Advanced"

        # Validate the list[cls] structure in the extracted data
        assert "related_items" in extracted_data
        assert isinstance(extracted_data["related_items"], list)
        assert len(extracted_data["related_items"]) >= 1
        assert "id" in extracted_data["related_items"][0]
        assert "name" in extracted_data["related_items"][0]
        assert "description" in extracted_data["related_items"][0]
        assert "is_active" in extracted_data["related_items"][0]

        # Log the extracted item for debugging
        self.log_extracted_items_for_instance(extracted_concept)

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(complex_concept)
        self.check_instance_serialization_and_cloning(document)

    def test_init_and_validate_date_concept(self):
        """
        Tests the initialization of the DateConcept class with valid and invalid input parameters.
        """

        # Valid initialization with default date format
        default_date_concept = DateConcept(
            name="Contract Date",
            description="Date when the contract was signed",
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            DateConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                extra_param=True,  # type: ignore
            )

        # Test attaching extracted items
        # Create a date string that matches the default format
        date_str = "02-03-2025"
        date_obj = default_date_concept._string_to_date(date_str)
        default_date_concept.extracted_items = [_DateItem(value=date_obj)]  # ty: ignore[invalid-assignment]

        # Test with invalid date string
        with pytest.raises(ValueError):
            date_str = "2025-01-01"
            date_obj = default_date_concept._string_to_date(date_str)

        # Test with invalid items
        with pytest.raises(ValueError):
            default_date_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _StringItem(value="15/01/2023")
            ]  # must be _DateItem

        with pytest.raises(ValueError):
            _DateItem(value="01-01-2025")  # must be a date object  # type: ignore

        # Test with justifications and references
        date_concept_with_refs = DateConcept(
            name="Date with refs",
            description="Test date concept with references",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(default_date_concept)
        self.check_instance_serialization_and_cloning(date_concept_with_refs)

    def test_init_and_validate_label_concept(self):
        """
        Tests the initialization of the LabelConcept class with valid and invalid input parameters.
        """
        # Valid initialization with multi-class classification
        multi_class_concept = LabelConcept(
            name="Document Type",
            description="Classify the type of legal document",
            labels=["NDA", "Consultancy Agreement", "Privacy Policy", "Other"],
            classification_type="multi_class",
            add_justifications=True,
        )

        # Valid initialization with multi-label classification
        multi_label_concept = LabelConcept(
            name="Content Topics",
            description="Identify all relevant topics covered in the document",
            labels=["Finance", "Legal", "Technology", "HR", "Operations", "Marketing"],
            classification_type="multi_label",
            model_role="reasoner",
            add_references=True,
            reference_depth="sentences",
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            LabelConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                labels=["A", "B", "C"],
                extra_param=True,  # type: ignore
            )

        # Test with insufficient labels (less than 2)
        with pytest.raises(ValueError):
            LabelConcept(
                name="Too Few Labels",
                description="Test with only one label",
                labels=["OnlyOne"],
            )

        # Test with duplicate labels
        with pytest.raises(ValueError):
            LabelConcept(
                name="Duplicate Labels",
                description="Test with duplicate labels",
                labels=["A", "B", "a", "C"],  # duplicate labels (case-insensitive)
            )

        # Test with invalid classification type
        with pytest.raises(ValueError):
            LabelConcept(
                name="Invalid Classification",
                description="Test invalid classification type",
                labels=["A", "B", "C"],
                classification_type="invalid_type",  # type: ignore
            )

        # Test with invalid label types (non-string types in labels list)
        with pytest.raises(ValueError):
            LabelConcept(
                name="Invalid Label Types",
                description="Test with non-string label types",
                labels=["A", 123, "C"],  # Integer in labels list  # type: ignore
            )

        # Test attaching extracted items for multi-class
        multi_class_item = _LabelItem(value=["NDA"])
        multi_class_concept.extracted_items = [multi_class_item]

        # Test attaching extracted items for multi-label
        multi_label_item = _LabelItem(value=["Finance", "Legal"])
        multi_label_concept.extracted_items = [multi_label_item]

        # Test with invalid items (wrong type)
        with pytest.raises(ValueError):
            multi_class_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _StringItem(value="NDA")
            ]  # must be _LabelItem

        # Test with invalid labels in extracted item
        with pytest.raises(ValueError):
            multi_class_concept.extracted_items = [
                _LabelItem(value=["InvalidLabel"])
            ]  # label not in predefined set

        # Test multi-class constraint validation (only one label allowed)
        with pytest.raises(ValueError):
            multi_class_concept._process_item_value({"labels": ["NDA", "Other"]})

        # Test multi-label allows multiple labels
        processed_labels = multi_label_concept._process_item_value(
            {"labels": ["Finance", "Legal", "Technology"]}
        )
        assert processed_labels == ["Finance", "Legal", "Technology"]

        # Test _format_labels_in_prompt property
        assert (
            multi_class_concept._format_labels_in_prompt
            == '["NDA", "Consultancy Agreement", "Privacy Policy", "Other"]'
        )
        assert (
            multi_label_concept._format_labels_in_prompt
            == '["Finance", "Legal", "Technology", "HR", "Operations", "Marketing"]'
        )

        # Test _item_type_in_prompt property
        assert multi_class_concept._item_type_in_prompt == "dict"
        assert multi_label_concept._item_type_in_prompt == "dict"

        # Test with justifications and references
        concept_with_refs = LabelConcept(
            name="Concept with refs",
            description="Test concept with references",
            labels=["Category1", "Category2", "Category3"],
            classification_type="multi_label",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
            singular_occurrence=True,
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(multi_class_concept)
        self.check_instance_serialization_and_cloning(multi_label_concept)
        self.check_instance_serialization_and_cloning(concept_with_refs)

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_extract_label_concept(self, llm: DocumentModelGroup | DocumentLLM):
        """
        Tests for label concept extraction from document using LLMs.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        # Create a document with content suitable for label classification
        document = Document(raw_text=get_test_document_text())

        # Test multi-class classification
        document_type_concept = LabelConcept(
            name="Document Type",
            description="Classify the type of legal document",
            labels=[
                "NDA",
                "Consultancy Agreement",
                "Privacy Policy",
                "Employment Contract",
                "Other",
            ],
            classification_type="multi_class",
            add_justifications=True,
            add_references=True,
            reference_depth="paragraphs",
            singular_occurrence=True,
        )

        # Test multi-label classification
        content_topics_concept = LabelConcept(
            name="Content Topics",
            description="Identify all relevant topics covered in this legal document",
            labels=[
                "Confidentiality",
                "Technology",
                "Finance",
                "Legal Compliance",
                "Third Party Relations",
                "Contract Terms",
            ],
            classification_type="multi_label",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Assign concepts to document
        document_concepts = [document_type_concept, content_topics_concept]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(2 if llm.is_group else 2),
            expected_n_calls_with_concurrency=(2 if llm.is_group else 2),
            expected_n_calls_1_item_per_call=(2 if llm.is_group else 2),
            func=llm.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )
        self.check_extra_data_in_extracted_items(document)

        # Verify extraction results
        extracted_document_type = document.get_concept_by_name("Document Type")
        extracted_content_topics = document.get_concept_by_name("Content Topics")

        # Check that items were extracted
        assert extracted_document_type.extracted_items
        assert extracted_content_topics.extracted_items

        # Verify multi-class constraint (only one label)
        document_type_item = extracted_document_type.extracted_items[0]
        assert len(document_type_item.value) == 1
        assert document_type_item.value[0] in document_type_concept.labels

        # Verify multi-label allows multiple labels
        content_topics_item = extracted_content_topics.extracted_items[0]
        assert len(content_topics_item.value) >= 1
        for label in content_topics_item.value:
            assert label in content_topics_concept.labels

        # Verify justifications and references are present
        assert document_type_item.justification
        assert document_type_item.reference_paragraphs
        assert content_topics_item.justification
        assert content_topics_item.reference_sentences

        # Verify that extraction references resolve to positions in the document,
        # in document order
        ref_para_indices = [
            document.get_paragraph_index(p)
            for p in document_type_item.reference_paragraphs
        ]
        assert ref_para_indices == sorted(ref_para_indices)
        ref_sent_positions = [
            document.get_sentence_index(s)
            for s in content_topics_item.reference_sentences
        ]
        assert ref_sent_positions == sorted(ref_sent_positions)

        # Log extracted items for verification
        self.log_extracted_items_for_instance(extracted_document_type)
        self.log_extracted_items_for_instance(extracted_content_topics)

        # Test scenario where no labels apply
        # Note: The framework allows empty extracted_items when multi-label classification
        # is used and no predefined labels apply. This is the expected behavior
        # as documented in LabelConcept.
        irrelevant_concept = LabelConcept(
            name="Technical Specifications",
            description="Identify technical specifications mentioned in the document",
            labels=[
                "Cooking Recipes",
                "Nutrition Information",
            ],
            classification_type="multi_label",
        )

        document.add_concepts([irrelevant_concept])
        llm.extract_concepts_from_document(
            document,
            from_concepts=[document.get_concept_by_name("Technical Specifications")],
        )

        # Should have empty extracted_items when no labels apply
        assert not document.get_concept_by_name(
            "Technical Specifications"
        ).extracted_items

        # For multi-class classification, a label is always returned, even if
        # it does not perfectly fit the content. This is the expected behavior
        # as documented in LabelConcept.
        irrelevant_concept = LabelConcept(
            name="Technical Specifications",
            description="Identify technical specifications mentioned in the document",
            labels=[
                "Cooking Recipes",
                "Nutrition Information",
            ],
            classification_type="multi_class",
        )
        document.concepts = [irrelevant_concept]
        llm.extract_concepts_from_document(document)
        technical_specifications_concept = document.get_concept_by_name(
            "Technical Specifications"
        )
        assert technical_specifications_concept.extracted_items
        assert (
            len(technical_specifications_concept.extracted_items[0].value) == 1
        )  # should always return a single label
        assert (
            technical_specifications_concept.extracted_items[0].value[0]
            in irrelevant_concept.labels
        )

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_concepts_from_document(document)

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(document)

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

    def test_init_example(self):
        """
        Tests the initialization of the example classes.
        """
        example = StringExample(content="Test")
        StringConcept(
            name="test",
            description="test",
            examples=[example],
        )
        with pytest.raises(ValueError):
            # Examples with same unique ID
            StringConcept(
                name="test",
                description="test",
                examples=[example, example],
            )
        with pytest.raises(ValueError):
            StringExample(content=1)  # type: ignore
        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(example)

        example = JsonObjectExample(content={"category": "test", "description": "test"})
        JsonObjectConcept(
            name="test",
            description="test",
            structure={"category": str, "description": str},
            examples=[example],
        )
        with pytest.raises(ValueError):
            # Examples with same unique ID
            JsonObjectConcept(
                name="test",
                structure={"category": str, "description": str},
                description="test",
                examples=[example, example],
            )
        with pytest.raises(ValueError):
            # Example structure does not align with the expected structure of the concept
            JsonObjectConcept(
                name="test",
                structure={"category": int, "value": float},
                description="test",
                examples=[example],
            )
        with pytest.raises(ValueError):
            JsonObjectExample(content={"category": str, "valid": bool})
        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(example)

    def test_init_item(self):
        """
        Tests the initialization of the extracted item classes.
        """
        _StringItem(value="Random string")
        _BooleanItem(value=True)
        _IntegerItem(value=1)
        _FloatItem(value=1.0)
        _IntegerOrFloatItem(value=1)
        _IntegerOrFloatItem(value=1.0)
        _JsonObjectItem(value={"hello": "world"})
        _LabelItem(value=["NDA"])
        with pytest.raises(TypeError):
            _ExtractedItem(value=1)
        with pytest.raises(ValueError):
            _StringItem()
        with pytest.raises(ValueError):
            _BooleanItem(value=1)  # type: ignore
        with pytest.raises(ValueError):
            _BooleanItem()
        with pytest.raises(ValueError):
            _BooleanItem(value="True")  # type: ignore
        with pytest.raises(ValueError):
            _IntegerOrFloatItem()
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=int)  # type: ignore
        with pytest.raises(ValueError):
            _JsonObjectItem()
        with pytest.raises(ValueError):
            _JsonObjectItem(value={})
        with pytest.raises(ValueError):
            _JsonObjectItem(value={"hello": {}})
        with pytest.raises(ValueError):
            _StringItem(
                value="Random string",
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            _LabelItem(value=[])
        with pytest.raises(ValueError):
            _LabelItem(value="NDA")  # type: ignore

        # List field items' unique IDs
        para = Paragraph(raw_text="Test")
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=1.0, reference_paragraphs=[para, para, para])  # type: ignore

        # Frozen state
        item = _IntegerOrFloatItem(value=1)
        with pytest.raises(ValueError):
            item.value = 2.0

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(item)

    def test_init_item_with_probabilities(self):
        """
        Tests the initialization of the extracted item classes with the probabilities
        of the options of the closed question that an item answers.
        """
        # No probabilities by default, which is the case for the items
        # extracted by an LLM
        for item in [
            _StringItem(value="Random string"),
            _BooleanItem(value=True),
            _IntegerItem(value=1),
            _FloatItem(value=1.0),
            _IntegerOrFloatItem(value=1),
            _JsonObjectItem(value={"hello": "world"}),
            _DateItem(value=date(2025, 3, 2)),
            _LabelItem(value=["NDA"]),
            _BooleanItem(value=True, justification="Random justification"),
        ]:
            assert item.probabilities is None
            assert "probabilities" not in item.model_fields_set
        assert _BooleanItem(value=True, probabilities=None).probabilities is None

        # Yes/no question: `True` and `False` are the options
        boolean_probabilities: ItemProbabilities = {True: 0.83, False: 0.17}
        source_probabilities = dict(boolean_probabilities)
        boolean_item = _BooleanItem(value=True, probabilities=source_probabilities)
        self.check_item_probabilities(boolean_item, boolean_probabilities)
        # The item does not share the dict it was created with
        source_probabilities[True] = 0.5
        del source_probabilities[False]
        self.check_item_probabilities(boolean_item, boolean_probabilities)

        # Item of an aspect: whether the paragraph belongs to the aspect
        string_item = _StringItem(
            value="Any notice shall be given in writing.",
            probabilities={True: 0.6, False: 0.4},
        )
        self.check_item_probabilities(string_item, {True: 0.6, False: 0.4})

        # One label out of several: the labels are the options
        multi_class_probabilities: ItemProbabilities = {
            "written": 0.71,
            "email": 0.20,
            "any": 0.06,
            "not specified": 0.03,
        }
        multi_class_item = _LabelItem(
            value=["written"], probabilities=multi_class_probabilities
        )
        self.check_item_probabilities(multi_class_item, multi_class_probabilities)

        # Several labels: each label has its own probability, i.e. the probabilities
        # are not required to sum to 1
        multi_label_probabilities: ItemProbabilities = {
            "NDA": 0.97,
            "Confidential": 0.62,
            "Other": 0.08,
        }
        assert sum(multi_label_probabilities.values()) > 1
        multi_label_item = _LabelItem(
            value=["NDA", "Confidential"], probabilities=multi_label_probabilities
        )
        self.check_item_probabilities(multi_label_item, multi_label_probabilities)
        probabilities_with_other_sums: list[ItemProbabilities] = [
            {"NDA": 0.1, "Confidential": 0.2, "Other": 0.3},
            {"NDA": 0.0, "Confidential": 0.0, "Other": 0.0},
            {"NDA": 1.0, "Confidential": 1.0, "Other": 1.0},
        ]
        for probabilities in probabilities_with_other_sums:
            self.check_item_probabilities(
                _LabelItem(value=["NDA"], probabilities=probabilities), probabilities
            )

        # Rating: the integers of the rating scale are the options
        rating_probabilities: ItemProbabilities = {
            1: 0.0,
            2: 0.0,
            3: 0.15,
            4: 0.55,
            5: 0.30,
        }
        rating_item = _IntegerItem(value=4, probabilities=rating_probabilities)
        self.check_item_probabilities(rating_item, rating_probabilities)

        # The other item classes accept the probabilities in the same way
        other_item_classes_and_values: list[tuple[type[_ExtractedItem], Any]] = [
            (_FloatItem, 1.5),
            (_IntegerOrFloatItem, 1),
            (_IntegerOrFloatItem, 1.5),
            (_JsonObjectItem, {"hello": "world"}),
            (_DateItem, date(2025, 3, 2)),
        ]
        for item_class, value in other_item_classes_and_values:
            for probabilities in (
                boolean_probabilities,
                multi_class_probabilities,
                rating_probabilities,
            ):
                item = item_class(value=value, probabilities=probabilities)
                assert type(item) is item_class
                assert item.value == value
                self.check_item_probabilities(item, probabilities)

        # The limits of the range are valid probabilities, and an integer
        # is converted to a float
        self.check_item_probabilities(
            _BooleanItem(value=True, probabilities={True: 1.0, False: 0.0}),
            {True: 1.0, False: 0.0},
        )
        item = _BooleanItem(value=False, probabilities={True: 0, False: 1})
        assert item.probabilities is not None
        assert [type(probability) for probability in item.probabilities.values()] == [
            float,
            float,
        ]
        self.check_item_probabilities(item, {True: 0.0, False: 1.0})

        # A single option
        self.check_item_probabilities(
            _LabelItem(value=["NDA"], probabilities={"NDA": 0.97}), {"NDA": 0.97}
        )

        # The order of the options is kept as given
        probabilities_in_other_orders: list[ItemProbabilities] = [
            {False: 0.17, True: 0.83},
            {5: 0.30, 1: 0.0, 4: 0.55, 3: 0.15, 2: 0.0},
            {"not specified": 0.03, "written": 0.71, "any": 0.06, "email": 0.20},
        ]
        for probabilities in probabilities_in_other_orders:
            self.check_item_probabilities(
                _StringItem(value="Random string", probabilities=probabilities),
                probabilities,
            )

        # The probabilities are kept with the other data of the item
        item = _BooleanItem(
            value=True,
            justification="Random justification",
            probabilities={True: 0.83, False: 0.17},
            custom_data={"source": "test"},
        )
        assert item.value is True
        assert item.justification == "Random justification"
        assert item.custom_data == {"source": "test"}
        self.check_item_probabilities(item, {True: 0.83, False: 0.17})

        # Frozen state
        with pytest.raises(ValueError):
            boolean_item.probabilities = {True: 0.5, False: 0.5}
        with pytest.raises(ValueError):
            boolean_item.probabilities = None
        self.check_item_probabilities(boolean_item, {True: 0.83, False: 0.17})
        item_without_probabilities = _BooleanItem(value=True)
        with pytest.raises(ValueError):
            item_without_probabilities.probabilities = {True: 0.5, False: 0.5}
        assert item_without_probabilities.probabilities is None

        # The items with probabilities are attached to the concepts and to the aspect
        # as the items without them
        boolean_concept = BooleanConcept(name="NDA check", description="Is an NDA")
        boolean_concept.extracted_items = [boolean_item]
        multi_class_concept = LabelConcept(
            name="Notice form",
            description="Form in which a notice is to be given",
            labels=["written", "email", "any", "not specified"],
            classification_type="multi_class",
        )
        multi_class_concept.extracted_items = [multi_class_item]
        multi_label_concept = LabelConcept(
            name="Document types",
            description="Types of the document",
            labels=["NDA", "Confidential", "Other"],
            classification_type="multi_label",
        )
        multi_label_concept.extracted_items = [multi_label_item]
        rating_concept = RatingConcept(
            name="Clarity", description="Clarity of the text", rating_scale=(1, 5)
        )
        rating_concept.extracted_items = [rating_item]
        aspect = Aspect(name="Notices", description="Clauses on notices")
        aspect.extracted_items = [string_item]
        instances_and_probabilities: list[
            tuple[Aspect | _Concept, ItemProbabilities]
        ] = [
            (boolean_concept, boolean_probabilities),
            (multi_class_concept, multi_class_probabilities),
            (multi_label_concept, multi_label_probabilities),
            (rating_concept, rating_probabilities),
            (aspect, {True: 0.6, False: 0.4}),
        ]
        for instance, expected_probabilities in instances_and_probabilities:
            assert len(instance.extracted_items) == 1
            self.check_item_probabilities(
                instance.extracted_items[0], expected_probabilities
            )

    def test_item_probabilities_option_types(self):
        """
        Tests that the options in the probabilities of an extracted item keep their
        exact types, i.e. that an option is never converted to an option of another
        type that is equal to it as a dict key (``True == 1``).
        """
        # Booleans are kept as booleans
        boolean_item = _BooleanItem(value=True, probabilities={True: 0.83, False: 0.17})
        assert boolean_item.probabilities is not None
        assert [type(option) for option in boolean_item.probabilities] == [bool, bool]
        assert list(boolean_item.probabilities) == [True, False]
        assert all(
            option is True or option is False for option in boolean_item.probabilities
        )

        # Integers are kept as integers, including the ones that are equal
        # to booleans, e.g. on a rating scale that starts at 0
        integer_item = _IntegerItem(value=1, probabilities={0: 0.25, 1: 0.75})
        assert integer_item.probabilities is not None
        assert [type(option) for option in integer_item.probabilities] == [int, int]
        self.check_item_probabilities(integer_item, {0: 0.25, 1: 0.75})
        # The dicts are equal, which is why the types are checked separately
        assert integer_item.probabilities == {False: 0.25, True: 0.75}
        with pytest.raises(AssertionError):
            self.check_item_probabilities(integer_item, {False: 0.25, True: 0.75})

        # Strings are kept as strings, including the ones that read as other options
        string_options: ItemProbabilities = {
            "1": 0.4,
            "0": 0.3,
            "True": 0.2,
            "false": 0.1,
            " a ": 0.0,
        }
        label_item = _LabelItem(value=["1"], probabilities=string_options)
        assert label_item.probabilities is not None
        assert [type(option) for option in label_item.probabilities] == [str] * 5
        self.check_item_probabilities(label_item, string_options)

        # Options of different types are kept apart in the same dict
        mixed_options: ItemProbabilities = {
            True: 0.5,
            "True": 0.3,
            2: 0.2,
            "2": 0.0,
            0: 0.0,
        }
        mixed_item = _StringItem(value="Random string", probabilities=mixed_options)
        assert mixed_item.probabilities is not None
        assert [type(option) for option in mixed_item.probabilities] == [
            bool,
            str,
            int,
            str,
            int,
        ]
        self.check_item_probabilities(mixed_item, mixed_options)

        # Large and negative integers are valid options
        integer_options: ItemProbabilities = {-1: 0.5, 10**30: 0.5}
        self.check_item_probabilities(
            _IntegerItem(value=1, probabilities=integer_options), integer_options
        )

        # A value of another type is rejected and not converted to an option:
        # with a non-strict validation, `1.0` would become `True`
        for invalid_option in (1.0, 0.0, 2.5, None, b"written", ("NDA", "Other")):
            with pytest.raises(ValueError, match="probabilities"):
                _BooleanItem(value=True, probabilities={invalid_option: 0.5})  # type: ignore

    @pytest.mark.parametrize(
        "invalid_probabilities",
        [
            # Empty
            {},
            # Not a dict
            [[True, 0.83], [False, 0.17]],  # serialized form
            [(True, 0.83), (False, 0.17)],
            ((True, 0.83), (False, 0.17)),
            [True, False],
            "0.83",
            0.83,
            True,
            # Probability is not a number
            {True: True, False: False},
            {True: False},
            {True: "0.83", False: "0.17"},
            {True: None},
            {True: [0.83]},
            {True: {True: 0.83}},
            # Probability is not a finite number
            {True: float("nan")},
            {True: float("inf")},
            {True: float("-inf")},
            # Probability is outside of the range
            {True: -0.01},
            {True: 1.01},
            {True: -1},
            {True: 2},
            {True: 83.0, False: 17.0},  # percentages
            # One invalid probability among the valid ones
            {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.55, 5: 3},
            {"written": 0.71, "email": float("nan"), "any": 0.06},
            # Option of an invalid type
            {1.0: 0.5},
            {0.5: 0.5},
            {None: 0.5},
            {b"written": 0.5},
            {("NDA", "Other"): 0.5},
            {True: 0.5, 1.5: 0.5},
        ],
    )
    def test_validate_item_probabilities(self, invalid_probabilities: Any):
        """
        Tests that invalid probabilities of an extracted item are rejected.
        """
        item_classes_and_values: list[tuple[type[_ExtractedItem], Any]] = [
            (_BooleanItem, True),
            (_StringItem, "Random string"),
            (_IntegerItem, 1),
            (_LabelItem, ["NDA"]),
        ]
        for item_class, value in item_classes_and_values:
            with pytest.raises(ValueError, match="probabilities"):
                item_class(value=value, probabilities=invalid_probabilities)

    def test_item_probabilities_equality(self):
        """
        Tests that the probabilities of an extracted item take part in the comparison
        of the items, i.e. that two items that differ only in probabilities
        are not equal.
        """
        item = _BooleanItem(value=True, probabilities={True: 0.83, False: 0.17})

        def item_with_probabilities(
            serialized_probabilities: list[list[Any]] | None,
        ) -> _BooleanItem:
            """
            Returns a copy of the item that differs from it only in probabilities.

            Two items that are created separately are never equal, as they have
            different unique IDs, i.e. such a copy is created from the serialized
            item, which keeps the unique ID.

            :param serialized_probabilities: The probabilities of the copy, as the
                list of [option, probability] pairs that they are serialized to.
            :type serialized_probabilities: list[list[Any]] | None
            :return: The copy of the item with the given probabilities.
            :rtype: _BooleanItem
            """
            item_dict = item.to_dict()
            item_dict["probabilities"] = serialized_probabilities
            return _BooleanItem.from_dict(item_dict)

        # Same probabilities
        assert item.clone() == item
        assert item_with_probabilities([[True, 0.83], [False, 0.17]]) == item
        assert item == item_with_probabilities([[True, 0.83], [False, 0.17]])

        # Another probability of an option
        assert item_with_probabilities([[True, 0.82], [False, 0.17]]) != item
        assert item != item_with_probabilities([[True, 0.82], [False, 0.17]])

        # Other options
        assert item_with_probabilities([[True, 0.83]]) != item
        assert (
            item_with_probabilities([[True, 0.83], [False, 0.17], ["a", 0.0]]) != item
        )
        assert item_with_probabilities([["True", 0.83], ["False", 0.17]]) != item

        # No probabilities
        item_without_probabilities = item_with_probabilities(None)
        assert item_without_probabilities.probabilities is None
        assert item_without_probabilities != item
        assert item != item_without_probabilities
        assert item_without_probabilities == item_with_probabilities(None)
        assert item_without_probabilities.clone() == item_without_probabilities

        # Options of another type that are equal as dict keys (`True == 1`)
        item_with_integer_options = item_with_probabilities([[1, 0.83], [0, 0.17]])
        assert item_with_integer_options.probabilities == item.probabilities
        assert item_with_integer_options != item
        assert item != item_with_integer_options
        assert item_with_integer_options == item_with_probabilities(
            [[1, 0.83], [0, 0.17]]
        )
        assert item_with_probabilities([[True, 0.83], [0, 0.17]]) != item

        # The same options with the same probabilities in another order: the items
        # are equal, as the dicts are
        item_with_other_order = item_with_probabilities([[False, 0.17], [True, 0.83]])
        assert item_with_other_order.probabilities is not None
        assert list(item_with_other_order.probabilities) == [False, True]
        assert item_with_other_order == item

        # Separately created items are not equal with or without probabilities,
        # as they have different unique IDs
        assert _BooleanItem(value=True, probabilities={True: 0.83, False: 0.17}) != item
        assert _BooleanItem(value=True) != _BooleanItem(value=True)

        # Objects of other types
        assert item != {True: 0.83, False: 0.17}
        assert item != [[True, 0.83], [False, 0.17]]
        assert item != {True: 0.83, False: 0.17}
        assert (
            _StringItem(value="True", probabilities={True: 0.83, False: 0.17}) != item
        )
