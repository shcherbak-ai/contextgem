# ContextGem: StringConcept Extraction with Examples

import os

from contextgem import Document, DocumentLLM, StringConcept, StringExample


# Create a Document object from text
contract_text = """
SERVICE AGREEMENT
This Service Agreement (the "Agreement") is entered into as of January 15, 2025 by and between:
XYZ Innovations Inc., a Delaware corporation with offices at 123 Tech Avenue, San Francisco, CA 
("Provider"), and
Omega Enterprises LLC, a New York limited liability company with offices at 456 Business Plaza, 
New York, NY ("Customer").
"""
doc = Document(raw_text=contract_text)

# Create a StringConcept for extracting parties and their roles
parties_concept = StringConcept(
    name="Contract parties",
    description="Names of parties and their roles in the contract",
    examples=[
        StringExample(content="Acme Corporation (Supplier)"),
        StringExample(content="TechGroup Inc. (Client)"),
    ],  # add examples providing additional guidance to the LLM
)

# Attach the concept to the document
doc.add_concepts([parties_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
parties_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted parties and their roles
print("Extracted parties and roles:")
for item in parties_concept.extracted_items:
    print(f"- {item.value}")

# Expected output:
# - XYZ Innovations Inc. (Provider)
# - Omega Enterprises LLC (Customer)
