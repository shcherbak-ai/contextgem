# Quick Start Example - Extracting concept from a document with an image

import os
from pathlib import Path

from contextgem import Document, DocumentLLM, Image, NumericalConcept, image_to_base64

# Path adapted for testing
current_file = Path(__file__).resolve()
root_path = current_file.parents[4]
image_path = root_path / "tests" / "images" / "invoices" / "invoice.jpg"

# Create an image instance
doc_image = Image(mime_type="image/jpg", base64_data=image_to_base64(image_path))

# Example document instance holding only the image
doc = Document(
    images=[doc_image],  # may contain multiple images
)

# Define a concept to extract the invoice total amount
doc_concept = NumericalConcept(
    name="Invoice Total",
    description="The total amount to be paid as shown on the invoice",
    numeric_type="float",
    llm_role="extractor_vision",  # use vision model
)

# Add concept to the document
doc.add_concepts([doc_concept])
# (add more concepts to the document, if needed)

# Create an LLM for extraction
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # Using a model with vision capabilities
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),  # your API key
    role="extractor_vision",  # mark LLM as vision model
)

# Extract information from the document
extracted_concepts = llm.extract_concepts_from_document(doc)
# or use async version: await llm.extract_concepts_from_document_async(doc)

# Access extracted information
print("Invoice Total:")
print(extracted_concepts[0].extracted_items)  # extracted concept items
# or doc.get_concept_by_name("Invoice Total").extracted_items
