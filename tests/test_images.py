# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for images and vision.
"""

from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image as PILImage

from contextgem.internal.base.concepts import _Concept
from contextgem.public import (
    Aspect,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    Image,
    JsonObjectConcept,
    JsonObjectExample,
    NumericalConcept,
    StringConcept,
)
from contextgem.public.utils import create_image, image_to_base64
from tests.base import BaseTest
from tests.utils import get_project_root_path, get_test_document_text


class TestImages(BaseTest):
    """
    Tests for images, and for the extraction from images and multimodal documents.
    """

    # Shared objects passed to the `parametrize` decorators of the tests below.
    # The decorators are evaluated in the class body, where the attributes inherited
    # from the base class are not in scope, so the names are bound here.
    llm_extractor = BaseTest.llm_extractor
    llm_reasoner = BaseTest.llm_reasoner
    model_group = BaseTest.model_group
    test_img_png_invoice = BaseTest.test_img_png_invoice
    test_img_jpg_invoice = BaseTest.test_img_jpg_invoice
    test_img_webp_invoice = BaseTest.test_img_webp_invoice

    @pytest.mark.parametrize(
        "image", [test_img_png_invoice, test_img_jpg_invoice, test_img_webp_invoice]
    )
    def test_init_and_attach_image(self, image: Image):
        """
        Tests for constructing a Image instance and attaching it to a Document instance.
        """

        with pytest.raises(ValueError):
            Image()
        with pytest.raises(ValueError):
            Image(
                mime_type="image/weoighwe",  # invalid MIME type  # type: ignore
                base64_data="base 64 encoded string",
            )
        with pytest.raises(ValueError):
            Image(
                mime_type="image/png",
                base64_data="base 64 encoded string",
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        document = Document(images=[image])
        with pytest.raises(ValueError):
            Document(images=[image, image])  # duplicate images with same base64 string

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.parametrize(
        "test_image_path",
        [
            "tests/images/invoices/invoice.png",
            "tests/images/invoices/invoice.jpg",
            "tests/images/invoices/invoice.webp",
        ],
    )
    def test_create_image(self, test_image_path: str):
        """
        Tests for the `create_image` utility function with various input types.
        """
        project_root = get_project_root_path()
        full_image_path = project_root / test_image_path

        # Ensure test image exists
        assert full_image_path.exists(), f"Test image not found: {full_image_path}"

        # Test 1: Create Image from file path (string)
        img_from_str_path = create_image(str(full_image_path))
        assert isinstance(img_from_str_path, Image)
        assert img_from_str_path.base64_data
        assert img_from_str_path.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 2: Create Image from file path (Path object)
        img_from_path_obj = create_image(full_image_path)
        assert isinstance(img_from_path_obj, Image)
        assert img_from_path_obj.base64_data
        assert img_from_path_obj.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 3: Create Image from PIL Image object
        pil_image = PILImage.open(full_image_path)
        img_from_pil = create_image(pil_image)
        assert isinstance(img_from_pil, Image)
        assert img_from_pil.base64_data
        assert img_from_pil.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 4: Create Image from file handle
        with open(full_image_path, "rb") as f:
            img_from_file_handle = create_image(f)
            assert isinstance(img_from_file_handle, Image)
            assert img_from_file_handle.base64_data
            assert img_from_file_handle.mime_type in [
                "image/png",
                "image/jpeg",
                "image/webp",
            ]

        # Test 5: Create Image from raw bytes
        with open(full_image_path, "rb") as f:
            image_bytes = f.read()
        img_from_bytes = create_image(image_bytes)
        assert isinstance(img_from_bytes, Image)
        assert img_from_bytes.base64_data
        assert img_from_bytes.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 6: Create Image from BytesIO
        buffer = BytesIO(image_bytes)
        img_from_bytesio = create_image(buffer)
        assert isinstance(img_from_bytesio, Image)
        assert img_from_bytesio.base64_data
        assert img_from_bytesio.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 7: Verify all methods produce equivalent results
        # (they should have the same MIME type and base64 data)
        expected_mime_type = img_from_str_path.mime_type
        assert img_from_path_obj.mime_type == expected_mime_type
        assert img_from_pil.mime_type == expected_mime_type
        assert img_from_file_handle.mime_type == expected_mime_type
        assert img_from_bytes.mime_type == expected_mime_type
        assert img_from_bytesio.mime_type == expected_mime_type

        # All should have same base64 data
        expected_base64 = img_from_str_path.base64_data
        assert img_from_path_obj.base64_data == expected_base64
        # assert img_from_pil.base64_data == expected_base64  # ignore due to PIL compression
        assert img_from_file_handle.base64_data == expected_base64
        assert img_from_bytes.base64_data == expected_base64
        assert img_from_bytesio.base64_data == expected_base64

        # Test error scenarios
        # Test 1: Non-existent file path
        with pytest.raises(FileNotFoundError):
            create_image("non_existent_image.jpg")

        # Test 2: PIL Image without format
        pil_image_no_format = PILImage.new("RGB", (100, 100), color="red")
        with pytest.raises(ValueError, match="Cannot determine image format"):
            create_image(pil_image_no_format)

        # Test 3: Invalid bytes data
        with pytest.raises(OSError, match="Cannot open image from bytes data"):
            create_image(b"invalid image data")

        # Test 4: Invalid file-like object
        invalid_buffer = BytesIO(b"not an image")
        with pytest.raises(OSError, match="Cannot open image from file-like object"):
            create_image(invalid_buffer)

    @pytest.mark.parametrize(
        "test_image_path",
        [
            "tests/images/invoices/invoice.png",
            "tests/images/invoices/invoice.jpg",
            "tests/images/invoices/invoice.webp",
        ],
    )
    def test_image_to_base64(self, test_image_path: str):
        """
        Tests for the `image_to_base64` utility function with various input types.
        """
        project_root = get_project_root_path()
        full_image_path = project_root / test_image_path

        # Ensure test image exists
        assert full_image_path.exists(), f"Test image not found: {full_image_path}"

        # Test 1: Convert from file path (string)
        base64_from_str_path = image_to_base64(str(full_image_path))
        assert isinstance(base64_from_str_path, str)

        # Test 2: Convert from file path (Path object)
        base64_from_path_obj = image_to_base64(full_image_path)
        assert isinstance(base64_from_path_obj, str)

        # Test 3: Convert from file handle
        with open(full_image_path, "rb") as f:
            base64_from_file_handle = image_to_base64(f)
            assert isinstance(base64_from_file_handle, str)

        # Test 4: Convert from raw bytes
        with open(full_image_path, "rb") as f:
            image_bytes = f.read()
        base64_from_bytes = image_to_base64(image_bytes)
        assert isinstance(base64_from_bytes, str)

        # Test 5: Convert from BytesIO
        buffer = BytesIO(image_bytes)
        base64_from_bytesio = image_to_base64(buffer)
        assert isinstance(base64_from_bytesio, str)

        # Test 6: Verify all methods produce identical results
        # (they should all have the same base64 data for the same source)
        assert base64_from_str_path == base64_from_path_obj
        assert base64_from_str_path == base64_from_file_handle
        assert base64_from_str_path == base64_from_bytes
        assert base64_from_str_path == base64_from_bytesio

        # Test error scenarios
        # Test 1: Non-existent file path
        with pytest.raises(FileNotFoundError):
            image_to_base64("non_existent_image.jpg")

        # Test 2: Invalid file-like object that can't be read
        invalid_buffer = BytesIO(b"some data")
        invalid_buffer.close()  # Close the buffer to make it unreadable
        with pytest.raises(OSError, match="Cannot read from file-like object"):
            image_to_base64(invalid_buffer)

    @pytest.mark.vcr
    def test_vision(self):
        """
        Tests for data extraction from document images using vision API.
        """
        with pytest.raises(ValueError):
            Document(images=[])

        # Invoice
        document_concepts = [
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                modality="image",
                add_justifications=True,
            ),
            JsonObjectConcept(
                name="Service items",
                description="Service items, their quantity and price",
                structure={
                    "service_name": str,
                    "quantity": int,
                    "price": str,
                },
                examples=[
                    JsonObjectExample(
                        content={
                            "service_name": "Service 1",
                            "quantity": 1,
                            "price": "100.00 EUR",
                        }
                    )
                ],
                model_role="reasoner",
                modality="image",
                add_justifications=True,
            ),
            JsonObjectConcept(
                name="Invoice amounts",
                description="Invoice amounts as they appear in the invoice",
                structure={
                    "line": str,
                    "amount": str,
                },
                modality="image",
            ),
            NumericalConcept(
                name="Total invoice amount",
                description="Total amount of the invoice",
                modality="image",
                numeric_type="float",
                add_justifications=True,
            ),
        ]
        document = Document(
            images=[self.test_img_png_invoice], concepts=document_concepts
        )

        self.config_llm_async_limiter_for_mock_responses(self.model_group)

        self.compare_with_concurrent_execution(
            llm=self.model_group,
            expected_n_calls_no_concurrency=3,
            expected_n_calls_with_concurrency=4,
            expected_n_calls_1_item_per_call=4,
            func=self.model_group.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )

        # Other
        document_concepts = [
            NumericalConcept(
                name="Number of bedrooms",
                description="Number of bedrooms in the floor plan",
                model_role="reasoner",
                modality="image",
                numeric_type="int",
                add_justifications=True,
            ),
            NumericalConcept(
                name="Number of balconies",
                description="Number of balconies in the floor plan",
                model_role="reasoner",
                modality="image",
                numeric_type="int",
                add_justifications=True,
            ),
            StringConcept(
                name="Room names",
                description="Names of the rooms in the floor plan",
                modality="image",
            ),
        ]
        document = Document(
            images=[self.test_img_png_apt_plan], concepts=document_concepts
        )

        self.compare_with_concurrent_execution(
            llm=self.model_group,
            expected_n_calls_no_concurrency=2,
            expected_n_calls_with_concurrency=3,
            expected_n_calls_1_item_per_call=3,
            func=self.model_group.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )

        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Number of bedrooms")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Number of balconies")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Room names")
        )

        # Check usage tokens
        self.check_usage(self.model_group)
        # Check cost
        self.check_cost(self.model_group)

        # Test error when model is not vision-capable
        llm_not_vision_capable = DocumentLLM(
            model="cerebras/llama-3.3-70b",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        with pytest.raises(ValueError, match="does not support vision"):
            llm_not_vision_capable.chat(
                "What's the type of this document?", images=[self.test_img_png_invoice]
            )

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [llm_extractor, llm_reasoner, model_group])
    def test_multimodal_concepts(self, llm: DocumentLLM | DocumentModelGroup):
        """
        Tests for multimodal concept extraction from multimodal document (text + images).
        """

        # ===Text + images===
        # Clone for method-scoped state modification
        document_multimodal = self.document_multimodal.clone()
        document_multimodal.add_concepts(
            [
                StringConcept(
                    name="Invoice number",
                    description="Number of the invoice",
                    modality="multimodal",
                    add_references=True,
                    add_justifications=True,
                ),
                DateConcept(
                    name="Invoice date",
                    description="Date of the invoice",
                    modality="multimodal",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        with pytest.warns(
            UserWarning,
            match="When extracting from images, references will not be added",
        ):
            # Document has both text and images - warns that refs will be added only for text
            llm.extract_concepts_from_document(document_multimodal)
        concept_invoice_number = document_multimodal.get_concept_by_name(
            "Invoice number"
        )
        concept_invoice_date = document_multimodal.get_concept_by_name("Invoice date")
        if llm is not self.llm_reasoner:
            self.log_extracted_items_for_instance(concept_invoice_number)
            assert concept_invoice_number.extracted_items
            assert concept_invoice_number.extracted_items[0].reference_paragraphs
            assert concept_invoice_number.extracted_items[0].justification
            assert not all(
                i.reference_paragraphs for i in concept_invoice_number.extracted_items
            )  # refs will be added only for text content, not for images
            assert all(i.justification for i in concept_invoice_number.extracted_items)
        else:
            self.log_extracted_items_for_instance(concept_invoice_date)
            assert concept_invoice_date.extracted_items
            assert concept_invoice_date.extracted_items[0].reference_paragraphs
            assert concept_invoice_date.extracted_items[0].reference_sentences
            assert not all(
                i.reference_paragraphs for i in concept_invoice_date.extracted_items
            )  # refs will be added only for text content, not for images

        # Test together with aspects and other concepts routed to different LLMs
        if isinstance(llm, DocumentModelGroup):
            document_multimodal.add_aspects(
                [
                    Aspect(
                        name="Invoices",
                        description="Provisions on invoices in the contract",
                        concepts=[
                            NumericalConcept(
                                name="Invoice number",
                                description="Number of the invoice",
                            )
                        ],
                    )
                ]
            )
            llm.extract_aspects_from_document(document_multimodal)
            assert document_multimodal.aspects[0].extracted_items
            document_multimodal.add_concepts(
                [
                    StringConcept(
                        name="Document type",
                        description="Guess the type of the document",
                        model_role="reasoner",
                        add_justifications=True,
                    )
                ]
            )
            llm.extract_all(document_multimodal, overwrite_existing=True)
            document_type_concept = document_multimodal.get_concept_by_name(
                "Document type"
            )
            assert document_type_concept.extracted_items
            self.log_extracted_items_for_instance(document_type_concept)
            assert document_multimodal.aspects[0].concepts[0].extracted_items

        # ===Text only===
        document = Document(raw_text=get_test_document_text())
        document.add_concepts(
            [
                StringConcept(
                    name="Document type",
                    description="Type of the document",
                    modality="multimodal",
                )
            ]
        )
        llm.extract_concepts_from_document(document)
        document_type_concept = document.get_concept_by_name("Document type")
        assert document_type_concept.extracted_items

        # ===Images only===
        document = Document(images=[self.test_img_png_invoice])
        document.add_concepts(
            [
                StringConcept(
                    name="Invoice number",
                    description="Number of the invoice",
                    modality="multimodal",
                )
            ]
        )
        llm.extract_concepts_from_document(document)
        invoice_number_concept = document.get_concept_by_name("Invoice number")
        assert invoice_number_concept.extracted_items
