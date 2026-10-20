# ContextGem: Citing the Pages of Extracted Items

import os

from docling.document_converter import DocumentConverter

from contextgem import Aspect, Document, DocumentLLM, NumericalConcept


result = DocumentConverter().convert("path/to/service_agreement.pdf")
document = Document.from_docling(result.document)

document.aspects = [
    Aspect(
        name="Termination",
        description="Clauses on the term of the agreement and its termination",
    ),
]
document.concepts = [
    NumericalConcept(
        name="Monthly hosting fee",
        description="The monthly fee for hosting, in EUR",
        numeric_type="float",
        add_references=True,  # the paragraphs that the value is found in
    ),
]

llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
)
document = llm.extract_all(document)

# The pages of the paragraphs that an aspect or an extracted item refers to
termination = document.get_aspect_by_name("Termination")
print(f"Termination: pages {termination.reference_page_numbers}")
# Termination: pages [2]

fee = document.get_concept_by_name("Monthly hosting fee")
for item in fee.extracted_items:
    print(f"Monthly hosting fee: {item.value} (pages {item.reference_page_numbers})")
# Monthly hosting fee: 1200.0 (pages [1])
