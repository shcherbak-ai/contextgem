# ContextGem: DateConcept Extraction

import os

from contextgem import DateConcept, Document, DocumentLLM

# Create a Document object from text
doc = Document(
    raw_text="The research paper was published on March 15, 2025 and has been cited 42 times since."
)

# Define a DateConcept to extract the publication date
date_concept = DateConcept(
    name="Publication date",
    description="The date when the paper was published",
)

# Attach the concept to the document
doc.add_concepts([date_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
date_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted value
print(
    type(date_concept.extracted_items[0].value), date_concept.extracted_items[0].value
)
# Output: <class 'datetime.date'> 2025-03-15

# Or access the extracted value from the document object
print(
    type(doc.concepts[0].extracted_items[0].value),
    doc.concepts[0].extracted_items[0].value,
)
# Output: <class 'datetime.date'> 2025-03-15
