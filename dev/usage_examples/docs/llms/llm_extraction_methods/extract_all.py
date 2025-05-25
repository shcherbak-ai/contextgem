# ContextGem: Extracting All Aspects and Concepts from Document

import os

from contextgem import Aspect, Document, DocumentLLM, StringConcept

# Sample text content
text_content = """
John Smith is a 30-year-old software engineer working at TechCorp. 
He has 5 years of experience in Python development and leads a team of 8 developers.
His annual salary is $95,000 and he graduated from MIT with a Computer Science degree.
"""

# Create a Document object from text
doc = Document(raw_text=text_content)

# Define aspects and concepts directly on the document
doc.aspects = [
    Aspect(
        name="Professional Information",
        description="Information about the person's career, job, and work experience",
    )
]

doc.concepts = [
    StringConcept(
        name="Person name",
        description="Full name of the person",
    )
]

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract all aspects and concepts from the document
processed_doc = llm.extract_all(doc)

# Access extracted aspect information
aspect = processed_doc.aspects[0]
print(f"Aspect: {aspect.name}")
print(f"Extracted items: {[item.value for item in aspect.extracted_items]}")

# Access extracted concept information
concept = processed_doc.concepts[0]
print(f"Concept: {concept.name}")
print(f"Extracted value: {concept.extracted_items[0].value}")
