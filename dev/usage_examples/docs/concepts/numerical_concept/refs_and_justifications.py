# ContextGem: NumericalConcept Extraction with References and Justifications

import os

from contextgem import Document, DocumentLLM, NumericalConcept

# Document with values that require calculation/inference
report_text = """
Quarterly Sales Report - Q2 2023

Product A: Sold 450 units at $75 each
Product B: Sold 320 units at $125 each
Product C: Sold 180 units at $95 each

Marketing expenses: $28,500
Operating costs: $42,700
"""

# Create a Document from the text
doc = Document(raw_text=report_text)

# Create a NumericalConcept for total revenue
total_revenue_concept = NumericalConcept(
    name="Total quarterly revenue",
    description="The total revenue calculated by multiplying units sold by their price",
    add_justifications=True,
    justification_depth="comprehensive",  # Detailed justification to show calculation steps
    justification_max_sents=4,  # Maximum number of sentences for justification
    add_references=True,
    reference_depth="paragraphs",  # Reference specific paragraphs
    singular_occurrence=True,  # Ensure that the data is merged into a single item
)

# Attach the concept to the document
doc.add_concepts([total_revenue_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/o4-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept
total_revenue_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted inferred value with justification
print("Calculated total quarterly revenue:")
for item in total_revenue_concept.extracted_items:
    print(f"\nTotal Revenue: {item.value}")
    print(f"Calculation Justification: {item.justification}")
    print("Source references:")
    for para in item.reference_paragraphs:
        print(f"- {para.raw_text}")
