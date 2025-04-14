# Quick Start Example - Extracting anomalies from a document, with source references and justifications

import os

from contextgem import Document, DocumentLLM, StringConcept

# Sample document text (shortened for brevity)
doc = Document(
    raw_text=(
        "Consultancy Agreement\n"
        "This agreement between Company A (Supplier) and Company B (Customer)...\n"
        "The term of the agreement is 1 year from the Effective Date...\n"
        "The Supplier shall provide consultancy services as described in Annex 2...\n"
        "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
        "The purple elephant danced gracefully on the moon while eating ice cream.\n"  # 💎 anomaly
        "This agreement is governed by the laws of Norway...\n"
    ),
)

# Attach a document-level concept
doc.concepts = [
    StringConcept(
        name="Anomalies",  # in longer contexts, this concept is hard to capture with RAG
        description="Anomalies in the document",
        add_references=True,
        reference_depth="sentences",
        add_justifications=True,
        justification_depth="brief",
        # see the docs for more configuration options
    )
    # add more concepts to the document, if needed
    # see the docs for available concepts: StringConcept, JsonObjectConcept, etc.
]
# Or use `doc.add_concepts([...])`

# Define an LLM for extracting information from the document
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or another provider/LLM
    api_key=os.environ.get(
        "CONTEXTGEM_OPENAI_API_KEY"
    ),  # your API key for the LLM provider
    # see the docs for more configuration options
)

# Extract information from the document
doc = llm.extract_all(doc)  # or use async version `await llm.extract_all_async(doc)`

# Access extracted information in the document object
print(
    doc.concepts[0].extracted_items
)  # extracted items with references & justifications
# or `doc.get_concept_by_name("Anomalies").extracted_items`
