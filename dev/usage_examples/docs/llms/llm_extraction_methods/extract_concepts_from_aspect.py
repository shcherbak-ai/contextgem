# ContextGem: Extracting Concepts from Specific Aspects

import os

from contextgem import Aspect, Document, DocumentLLM, NumericalConcept, StringConcept

# Sample text content
text_content = """
DataFlow Systems is an innovative fintech startup that was established in 2020 in Austin, Texas.
The company has rapidly grown to 150 employees and operates in 8 major cities across North America.
DataFlow's core platform, FinanceStream, is used by more than 5,000 small businesses for automated accounting.
In their latest financial report, DataFlow Systems announced $12 million in annual revenue for 2024.
This represents an impressive 40% increase compared to their 2023 performance.
The company has secured $25 million in Series B funding and plans to expand internationally next year.
"""

# Create a Document object from text
doc = Document(raw_text=text_content)

# Define an aspect to extract from the document
financial_aspect = Aspect(
    name="Financial Performance",
    description="Revenue, growth metrics, and financial indicators",
)

# Add concepts to the aspect
financial_aspect.concepts = [
    StringConcept(
        name="Annual Revenue",
        description="Total revenue reported for the year",
    ),
    NumericalConcept(
        name="Growth Rate",
        description="Percentage growth rate compared to previous year",
        numeric_type="float",
    ),
    NumericalConcept(
        name="Revenue Year",
        description="The year for which revenue is reported",
    ),
]

# Attach the aspect to the document
doc.aspects = [financial_aspect]

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# First, extract the aspect from the document (required before concept extraction)
extracted_aspects = llm.extract_aspects_from_document(doc)
financial_aspect = extracted_aspects[0]

# Extract concepts from the specific aspect
extracted_concepts = llm.extract_concepts_from_aspect(financial_aspect, doc)

# Access extracted concepts for the aspect
print(f"Aspect: {financial_aspect.name}")
print(f"Extracted items: {[item.value for item in financial_aspect.extracted_items]}")
print("\nConcepts extracted from this aspect:")
for concept in extracted_concepts:
    print(f"  {concept.name}: {[item.value for item in concept.extracted_items]}")
