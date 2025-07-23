# ContextGem: Contract Type Classification using LabelConcept

import os

from contextgem import Document, DocumentLLM, LabelConcept


# Create a Document object from legal document text
legal_doc_text = """
NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into as of January 15, 2025, by and between TechCorp Inc., a Delaware corporation ("Disclosing Party"), and DataSystems LLC, a California limited liability company ("Receiving Party").

WHEREAS, Disclosing Party possesses certain confidential information relating to its proprietary technology and business operations;

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. CONFIDENTIAL INFORMATION
The term "Confidential Information" shall mean any and all non-public information...

2. OBLIGATIONS OF RECEIVING PARTY
Receiving Party agrees to hold all Confidential Information in strict confidence...
"""

doc = Document(raw_text=legal_doc_text)

# Define a LabelConcept for contract type classification
contract_type_concept = LabelConcept(
    name="Contract Type",
    description="Classify the type of contract",
    labels=["NDA", "Consultancy Agreement", "Privacy Policy", "Other"],
    classification_type="multi_class",  # only one label can be selected (mutually exclusive labels)
    singular_occurrence=True,  # expect only one classification result
)
print(contract_type_concept._format_labels_in_prompt)

# Attach the concept to the document
doc.add_concepts([contract_type_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
contract_type_concept = llm.extract_concepts_from_document(doc)[0]

# Check if any labels were extracted
if contract_type_concept.extracted_items:
    # Get the classified document type
    classified_type = contract_type_concept.extracted_items[0].value
    print(f"Document classified as: {classified_type}")  # Output: ['NDA']
else:
    print("No applicable labels found for this document")
