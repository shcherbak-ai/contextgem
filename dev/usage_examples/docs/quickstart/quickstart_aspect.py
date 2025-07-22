# Quick Start Example - Extracting aspect from a document

import os

from contextgem import Aspect, Document, DocumentLLM


# Example document instance
# Document content is shortened for brevity
doc = Document(
    raw_text=(
        "Consultancy Agreement\n"
        "This agreement between Company A (Supplier) and Company B (Customer)...\n"
        "The term of the agreement is 1 year from the Effective Date...\n"
        "The Supplier shall provide consultancy services as described in Annex 2...\n"
        "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
        "This agreement is governed by the laws of Norway...\n"
    ),
)

# Define an aspect with optional concept(s), using natural language
doc_aspect = Aspect(
    name="Governing law",
    description="Clauses defining the governing law of the agreement",
    reference_depth="sentences",
)

# Add aspects to the document
doc.add_aspects([doc_aspect])
# (add more aspects to the document, if needed)

# Create an LLM for extraction
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or any other LLM from e.g. Anthropic, etc.
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),  # your API key
)

# Extract information from the document
extracted_aspects = llm.extract_aspects_from_document(doc)
# or use async version llm.extract_aspects_from_document_async(doc)

# Access extracted information
print("Governing law aspect:")
print(
    extracted_aspects[0].extracted_items
)  # extracted aspect items with references to sentences
# or doc.get_aspect_by_name("Governing law").extracted_items
