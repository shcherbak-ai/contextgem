# ContextGem: BooleanConcept Extraction with References and Justifications

import os

from contextgem import BooleanConcept, Document, DocumentLLM


# Sample document text containing policy information
policy_text = """
Company Data Retention Policy (Updated 2024)

All customer data must be encrypted at rest and in transit using industry-standard encryption protocols.
Personal information should be retained for no longer than 3 years after the customer relationship ends.
Employees are required to complete data privacy training annually.
"""

# Create a Document from the text
doc = Document(raw_text=policy_text)

# Create a BooleanConcept with justifications and references enabled
compliance_concept = BooleanConcept(
    name="Has encryption requirement",
    description="Whether the document specifies that data must be encrypted",
    add_justifications=True,  # Enable justifications to understand reasoning
    justification_depth="brief",
    justification_max_sents=1,  # Allow up to 1 sentences for each justification
    add_references=True,  # Include references to source text
    reference_depth="sentences",  # Reference specific sentences rather than paragraphs
)

# Attach the concept to the document
doc.add_concepts([compliance_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4o-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept
compliance_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted value with justification and references
print(f"Has encryption requirement: {compliance_concept.extracted_items[0].value}")
print(f"\nJustification: {compliance_concept.extracted_items[0].justification}")
print("\nSource references:")
for sent in compliance_concept.extracted_items[0].reference_sentences:
    print(f"- {sent.raw_text}")
