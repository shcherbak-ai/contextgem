# ContextGem: DateConcept Extraction with References and Justifications

import os

from contextgem import DateConcept, Document, DocumentLLM


# Sample document text containing project timeline information
project_text = """
Project Timeline: Website Redesign

The website redesign project officially kicked off on March 1, 2024.
The development team has estimated the project will take 4 months to complete.

Key milestones:
- Design phase: 1 month
- Development phase: 2 months  
- Testing and deployment: 1 month

The marketing team needs the final completion date to plan the launch campaign.
"""

# Create a Document from the text
doc = Document(raw_text=project_text)

# Create a DateConcept to calculate the project completion date
completion_date_concept = DateConcept(
    name="Project completion date",
    description="The final completion date for the website redesign project",
    add_justifications=True,  # enable justifications to understand extraction logic
    justification_depth="balanced",
    justification_max_sents=3,  # allow up to 3 sentences for the calculation justification
    add_references=True,  # include references to source text
    reference_depth="sentences",  # reference specific sentences rather than paragraphs
    singular_occurrence=True,  # extract only one calculated date
)

# Attach the concept to the document
doc.add_concepts([completion_date_concept])

# Configure DocumentLLM
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept
completion_date_concept = llm.extract_concepts_from_document(doc)[0]

# Print the calculated completion date with justification and references
print("Calculated project completion date:")
extracted_item = completion_date_concept.extracted_items[
    0
]  # get the single calculated date
print(f"\nCompletion Date: {extracted_item.value}")  # expected output: 2024-07-01
print(f"Calculation Justification: {extracted_item.justification}")
print("Source references used for calculation:")
for sent in extracted_item.reference_sentences:
    print(f"- {sent.raw_text}")
