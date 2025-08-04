from pathlib import Path

from contextgem import Document, Paragraph, create_image


# Create a document with raw text content
contract_document = Document(
    raw_text=(
        "...This agreement is effective as of January 1, 2025.\n\n"
        "All parties must comply with the terms outlined herein. The terms include "
        "monthly reporting requirements and quarterly performance reviews.\n\n"
        "Failure to adhere to these terms may result in termination of the agreement. "
        "Additionally, any breach of confidentiality will be subject to penalties as "
        "described in this agreement.\n\n"
        "This agreement shall remain in force for a period of three (3) years unless "
        "otherwise terminated according to the provisions stated above..."
    ),
    paragraph_segmentation_mode="newlines",  # Default mode, splits on newlines
)

# Create a document with more advanced paragraph segmentation using a SaT model
report_document = Document(
    raw_text=(
        "Executive Summary "
        "This report outlines our quarterly performance. "
        "Revenue increased by [15%] compared to the previous quarter.\n\n"
        "Customer satisfaction metrics show positive trends across all regions..."
    ),
    paragraph_segmentation_mode="sat",  # Use SaT model for intelligent paragraph segmentation
    sat_model_id="sat-3l-sm",  # Specify which SaT model to use
)

# Create a document with predefined paragraphs, e.g. when you use a custom
# paragraph segmentation tool
document_from_paragraphs = Document(
    paragraphs=[
        Paragraph(raw_text="This is the first paragraph."),
        Paragraph(raw_text="This is the second paragraph with more content."),
        Paragraph(raw_text="Final paragraph concluding the document."),
        # ...
    ]
)

# Create document with images

# Path is adapted for doc tests
current_file = Path(__file__).resolve()
root_path = current_file.parents[4]
image_path = root_path / "tests" / "images" / "invoices" / "invoice.png"

# Create a document with only images (no text)
image_document = Document(
    images=[
        create_image(image_path),  # contextgem.Image instance
        # ...
    ]
)

# Create a document with both text and images
mixed_document = Document(
    raw_text="This document contains both text and visual elements.",
    images=[
        create_image(image_path),  # contextgem.Image instance
        # ...
    ],
)
