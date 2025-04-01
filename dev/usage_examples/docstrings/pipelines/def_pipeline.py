from contextgem import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentPipeline,
    StringConcept,
)

# Create a pipeline for NDA (Non-Disclosure Agreement) review
nda_pipeline = DocumentPipeline(
    aspects=[
        Aspect(
            name="Confidential information",
            description="Clauses defining the confidential information",
        ),
        Aspect(
            name="Exclusions",
            description="Clauses defining exclusions from confidential information",
        ),
        Aspect(
            name="Obligations",
            description="Clauses defining confidentiality obligations",
        ),
        Aspect(
            name="Liability",
            description="Clauses defining liability for breach of the agreement",
        ),
        # ... Add more aspects as needed
    ],
    concepts=[
        StringConcept(
            name="Anomaly",
            description="Anomaly in the contract, e.g. out-of-context or nonsensical clauses",
            llm_role="reasoner_text",
            add_references=True,  # Add references to the source text
            reference_depth="sentences",  # Reference to the sentence level
            add_justifications=True,  # Add justifications for the anomaly
            justification_depth="balanced",  # Justification at the sentence level
            justification_max_sents=5,  # Maximum number of sentences in the justification
        ),
        BooleanConcept(
            name="Is mutual",
            description="Whether the NDA is mutual (bidirectional) or one-way",
            singular_occurrence=True,
            llm_role="reasoner_text",  # Use the reasoner role for this concept
        ),
        DateConcept(
            name="Effective date",
            description="The date when the NDA agreement becomes effective",
            singular_occurrence=True,
        ),
        StringConcept(
            name="Term",
            description="The term of the NDA",
        ),
        StringConcept(
            name="Governing law",
            description="The governing law of the agreement",
            singular_occurrence=True,
        ),
        # ... Add more concepts as needed
    ],
)

# Assign the pipeline to the NDA document
nda_document = Document(raw_text="[NDA text]")
nda_document.assign_pipeline(nda_pipeline)

# Now the document is ready for processing with the NDA review pipeline!
# The document can be processed to extract the defined aspects and concepts

# Extract all aspects and concepts from the NDA using an LLM group
# with LLMs with roles "extractor_text" and "reasoner_text".
# llm_group.extract_all(nda_document)
