# ContextGem: NumericalConcept Extraction

import os

from contextgem import Document, DocumentLLM, NumericalConcept

# Create a Document object from text
doc = Document(
    raw_text="The latest smartphone model costs $899.99 and will be available next week."
)

# Define a NumericalConcept to extract the price
price_concept = NumericalConcept(
    name="Product price",
    description="The price of the product",
    numeric_type="float",  # We expect a decimal price
)

# Attach the concept to the document
doc.add_concepts([price_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
price_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted value
print(price_concept.extracted_items[0].value)  # Output: 899.99
# Or access the extracted value from the document object
print(doc.concepts[0].extracted_items[0].value)  # Output: 899.99
