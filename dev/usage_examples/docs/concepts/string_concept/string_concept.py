# ContextGem: StringConcept Extraction

import os

from contextgem import Document, DocumentLLM, StringConcept


# Create a Document object from text
doc = Document(raw_text="My name is John Smith and I am 30 years old.")

# Define a StringConcept to extract a person's name
name_concept = StringConcept(
    name="Person name",
    description="Full name of the person",
)

# Attach the concept to the document
doc.add_concepts([name_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
name_concept = llm.extract_concepts_from_document(doc)[0]

# Get the extracted value
print(name_concept.extracted_items[0].value)  # Output: "John Smith"
# Or access the extracted value from the document object
print(doc.concepts[0].extracted_items[0].value)  # Output: "John Smith"
