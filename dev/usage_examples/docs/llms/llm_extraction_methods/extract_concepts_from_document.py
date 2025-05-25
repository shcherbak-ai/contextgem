# ContextGem: Extracting Concepts Directly from Documents

import os

from contextgem import Document, DocumentLLM, NumericalConcept, StringConcept

# Sample text content
text_content = """
GreenTech Solutions is an environmental technology company founded in 2018 in Portland, Oregon.
The company develops sustainable energy solutions and has 75 employees working remotely across the United States.
Their primary product, EcoMonitor, helps businesses track carbon emissions and has been adopted by 2,500 organizations.
GreenTech Solutions reported strong financial performance with $8.5 million in revenue for 2024.
The company's CEO, Sarah Johnson, announced plans to achieve carbon neutrality by 2025.
They recently opened a new research facility in Seattle and hired 20 additional engineers.
"""

# Create a Document object from text
doc = Document(raw_text=text_content)

# Define concepts to extract from the document
doc.concepts = [
    StringConcept(
        name="Company Name",
        description="Full name of the company",
    ),
    StringConcept(
        name="CEO Name",
        description="Full name of the company's CEO",
    ),
    NumericalConcept(
        name="Employee Count",
        description="Total number of employees at the company",
        numeric_type="int",
    ),
    StringConcept(
        name="Annual Revenue",
        description="Company's total revenue for the year",
    ),
]

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract concepts from the document
extracted_concepts = llm.extract_concepts_from_document(doc)

# Access extracted concept information
print("Concepts extracted from document:")
for concept in extracted_concepts:
    print(f"  {concept.name}: {[item.value for item in concept.extracted_items]}")
