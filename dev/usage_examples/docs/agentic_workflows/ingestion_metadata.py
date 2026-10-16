# ContextGem: Extracting the Metadata of a Document at Ingestion

import os

from contextgem import (
    BooleanConcept,
    DateConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    ExtractionPipeline,
    LabelConcept,
    StringConcept,
)


# The metadata fields, defined once for all the documents
metadata_pipeline = ExtractionPipeline(
    concepts=[
        # Closed questions: answered by the decision model, with probabilities
        LabelConcept(
            name="Document type",
            description="Type of the document",
            labels=["NDA", "Service agreement", "Lease", "Other"],
            classification_type="multi_class",
            model_role="decide",
        ),
        BooleanConcept(
            name="Automatic renewal",
            description="The agreement renews automatically",
            model_role="decide",
        ),
        # Open values: extracted by the LLM
        StringConcept(
            name="Parties",
            description="Names of the parties to the document",
        ),
        DateConcept(
            name="Effective date",
            description="Date from which the document is in effect",
        ),
    ]
)

model_group = DocumentModelGroup(
    models=[
        DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        ),
        DocumentDecisionModel(
            model="typesafe/jev-latest",
            api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
            role="decide",
        ),
    ]
)

# A new document comes in
document = Document(
    raw_text=(
        "Service Agreement\n"
        "This agreement is made on 1 March 2026 between Acme Ltd (Customer) "
        "and Northwind AS (Supplier).\n"
        "The Supplier shall provide cloud hosting services as described "
        "in Schedule 1.\n"
        "This agreement renews automatically for successive one-year terms.\n"
        "Either party may terminate this agreement for any reason by giving "
        "30 days' written notice to the other party.\n"
        "This agreement is governed by the laws of Norway.\n"
    ),
)
document.assign_pipeline(metadata_pipeline)  # each document gets a copy
document = model_group.extract_all(document)

doc_type = document.get_concept_by_name("Document type").extracted_items[0]
label = doc_type.value[0]
renewal = document.get_concept_by_name("Automatic renewal").extracted_items[0]
parties = document.get_concept_by_name("Parties").extracted_items
dates = document.get_concept_by_name("Effective date").extracted_items

# The record of the document, for your search index or database
record = {
    "id": "acme-service-agreement",
    "document_type": label,  # "Service agreement"
    "document_type_probability": doc_type.probabilities[label],  # 1.0
    "automatic_renewal": renewal.value,  # True, with a probability of 0.99
    "parties": [p.value for p in parties],  # ["Acme Ltd", "Northwind AS"]
    "effective_date": [d.value.isoformat() for d in dates],  # ["2026-03-01"]
    # The document with its extracted items, which `Document.from_json()`
    # loads later without extracting again
    "contextgem_document": document.to_json(),
}
