# Quick Start Example - Extracting a concept from a document

import os

from contextgem import Document, DocumentLLM, JsonObjectConcept


# Example document instance
# Document content is shortened for brevity
doc = Document(
    raw_text=(
        "Statement of Work\n"
        "Project: Cloud Migration Initiative\n"
        "Client: Acme Corporation\n"
        "Contractor: TechSolutions Inc.\n\n"
        "Project Timeline:\n"
        "Start Date: March 1, 2025\n"
        "End Date: August 31, 2025\n\n"
        "Deliverables:\n"
        "1. Infrastructure assessment report (Due: March 15, 2025)\n"
        "2. Migration strategy document (Due: April 10, 2025)\n"
        "3. Test environment setup (Due: May 20, 2025)\n"
        "4. Production migration (Due: July 15, 2025)\n"
        "5. Post-migration support (Due: August 31, 2025)\n\n"
        "Budget: $250,000\n"
        "Payment Schedule: 20% upfront, 30% at midpoint, 50% upon completion\n"
    ),
)

# Define a document-level concept using e.g. JsonObjectConcept
# This will extract structured data from the entire document
doc_concept = JsonObjectConcept(
    name="Project Details",
    description="Key project information including timeline, deliverables, and budget",
    structure={
        "project_name": str,
        "client": str,
        "contractor": str,
        "budget": str,
        "payment_terms": str,
    },  # simply use a dictionary with type hints (including generic aliases and union types)
    add_references=True,
    reference_depth="paragraphs",
)

# Add the concept to the document
doc.add_concepts([doc_concept])
# (add more concepts to the document, if needed)

# Create an LLM for extraction
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or any other LLM from e.g. Anthropic, etc.
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),  # your API key
)

# Extract information from the document
extracted_concepts = llm.extract_concepts_from_document(doc)
# or use async version llm.extract_concepts_from_document_async(doc)

# Access extracted information
print("Project Details:")
print(
    extracted_concepts[0].extracted_items
)  # extracted concept items with references to paragraphs
# Or doc.get_concept_by_name("Project Details").extracted_items
