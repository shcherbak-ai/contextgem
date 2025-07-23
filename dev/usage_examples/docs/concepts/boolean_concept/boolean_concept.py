# ContextGem: BooleanConcept Extraction

import os

from contextgem import BooleanConcept, Document, DocumentLLM


# Create a Document object from text
doc = Document(
    raw_text="This document contains confidential information and should not be shared publicly."
)

# Define a BooleanConcept to detect confidential content
confidentiality_concept = BooleanConcept(
    name="Is confidential",
    description="Whether the document contains confidential information",
)

# Attach the concept to the document
doc.add_concepts([confidentiality_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
confidentiality_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted value
print(confidentiality_concept.extracted_items[0].value)  # Output: True
# Or access the extracted value from the document object
print(doc.concepts[0].extracted_items[0].value)  # Output: True
