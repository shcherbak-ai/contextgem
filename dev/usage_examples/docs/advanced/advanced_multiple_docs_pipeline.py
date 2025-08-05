# Advanced Usage Example - analyzing multiple documents with a single pipeline,
# with different LLMs, concurrency and cost tracking

import os

from contextgem import (
    Aspect,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentLLMGroup,
    ExtractionPipeline,
    JsonObjectConcept,
    JsonObjectExample,
    LLMPricing,
    NumericalConcept,
    RatingConcept,
    StringConcept,
    StringExample,
)


# Construct documents

# Document 1 - Consultancy Agreement (shortened for brevity)
doc1 = Document(
    raw_text=(
        "Consultancy Agreement\n"
        "This agreement between Company A (Supplier) and Company B (Customer)...\n"
        "The term of the agreement is 1 year from the Effective Date...\n"
        "The Supplier shall provide consultancy services as described in Annex 2...\n"
        "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
        "All intellectual property created during the provision of services shall belong to the Customer...\n"
        "This agreement is governed by the laws of Norway...\n"
        "Annex 1: Data processing agreement...\n"
        "Annex 2: Statement of Work...\n"
        "Annex 3: Service Level Agreement...\n"
    ),
)

# Document 2 - Service Level Agreement (shortened for brevity)
doc2 = Document(
    raw_text=(
        "Service Level Agreement\n"
        "This agreement between TechCorp (Provider) and GlobalInc (Client)...\n"
        "The agreement shall commence on January 1, 2023 and continue for 2 years...\n"
        "The Provider shall deliver IT support services as outlined in Schedule A...\n"
        "The Client shall make monthly payments of $5,000 within 15 days of invoice receipt...\n"
        "The Provider guarantees [99.9%] uptime for all critical systems...\n"
        "Either party may terminate with 60 days written notice...\n"
        "This agreement is governed by the laws of California...\n"
        "Schedule A: Service Descriptions...\n"
        "Schedule B: Response Time Requirements...\n"
    ),
)

# Create a reusable extraction pipeline
contract_pipeline = ExtractionPipeline()

# Define aspects and aspect-level concepts in the pipeline
# Concepts in the aspects will be extracted from the extracted aspect context
contract_pipeline.aspects = [  # or use .add_aspects([...])
    Aspect(
        name="Contract Parties",
        description="Clauses defining the parties to the agreement",
        concepts=[  # define aspect-level concepts, if any
            StringConcept(
                name="Party names and roles",
                description="Names of all parties entering into the agreement and their roles",
                examples=[  # optional
                    StringExample(
                        content="X (Client)",  # guidance regarding the expected output format
                    )
                ],
            )
        ],
    ),
    Aspect(
        name="Term",
        description="Clauses defining the term of the agreement",
        concepts=[
            NumericalConcept(
                name="Contract term",
                description="The term of the agreement in years",
                numeric_type="int",  # or "float", or "any" for auto-detection
                add_references=True,  # extract references to the source text
                reference_depth="paragraphs",
            )
        ],
    ),
]

# Define document-level concepts
# Concepts in the document will be extracted from the whole document content
contract_pipeline.concepts = [  # or use .add_concepts()
    DateConcept(
        name="Effective date",
        description="The effective date of the agreement",
    ),
    StringConcept(
        name="Contract type",
        description="The type of agreement",
        llm_role="reasoner_text",  # for this concept, we use a more advanced LLM for reasoning
    ),
    StringConcept(
        name="Governing law",
        description="The law that governs the agreement",
    ),
    JsonObjectConcept(
        name="Attachments",
        description="The titles and concise descriptions of the attachments to the agreement",
        structure={"title": str, "description": str | None},
        examples=[  # optional
            JsonObjectExample(  # guidance regarding the expected output format
                content={
                    "title": "Appendix A",
                    "description": "Code of conduct",
                }
            ),
        ],
    ),
    RatingConcept(
        name="Duration adequacy",
        description="Contract duration adequacy considering the subject matter and best practices.",
        llm_role="reasoner_text",  # for this concept, we use a more advanced LLM for reasoning
        rating_scale=(1, 10),
        add_justifications=True,  # add justifications for the rating
        justification_depth="balanced",  # provide a balanced justification
        justification_max_sents=3,
    ),
]

# Assign pipeline to the documents
# You can re-use the same pipeline for multiple documents
doc1.assign_pipeline(
    contract_pipeline
)  # assigns pipeline aspects and concepts to the document
doc2.assign_pipeline(
    contract_pipeline
)  # assigns pipeline aspects and concepts to the document

# Create an LLM group for data extraction and reasoning
llm_extractor = DocumentLLM(
    model="openai/gpt-4o-mini",  # or any other LLM from e.g. Anthropic, etc.
    api_key=os.environ["CONTEXTGEM_OPENAI_API_KEY"],  # your API key
    role="extractor_text",  # signifies the LLM is used for data extraction tasks
    pricing_details=LLMPricing(  # optional, for costs calculation
        input_per_1m_tokens=0.150,
        output_per_1m_tokens=0.600,
    ),
)
llm_reasoner = DocumentLLM(
    model="openai/o3-mini",  # or any other LLM from e.g. Anthropic, etc.
    api_key=os.environ["CONTEXTGEM_OPENAI_API_KEY"],  # your API key
    role="reasoner_text",  # signifies the LLM is used for reasoning tasks
    pricing_details=LLMPricing(  # optional, for costs calculation
        input_per_1m_tokens=1.10,
        output_per_1m_tokens=4.40,
    ),
)
# The LLM group is used for all extraction tasks within the pipeline
llm_group = DocumentLLMGroup(llms=[llm_extractor, llm_reasoner])

# Extract all information from the documents at once
doc1 = llm_group.extract_all(
    doc1, use_concurrency=True
)  # use concurrency to speed up extraction
doc2 = llm_group.extract_all(
    doc2, use_concurrency=True
)  # use concurrency to speed up extraction
# Or use async variants .extract_all_async(...)

# Get the extracted data
print("Some extracted data from doc 1:")
print("Contract Parties > Party names and roles:")
print(
    doc1.get_aspect_by_name("Contract Parties")
    .get_concept_by_name("Party names and roles")
    .extracted_items
)
print("Attachments:")
print(doc1.get_concept_by_name("Attachments").extracted_items)
# ...

print("\nSome extracted data from doc 2:")
print("Term > Contract term:")
print(
    doc2.get_aspect_by_name("Term")
    .get_concept_by_name("Contract term")
    .extracted_items[0]
    .value
)
print("Duration adequacy:")
print(doc2.get_concept_by_name("Duration adequacy").extracted_items[0].value)
print(doc2.get_concept_by_name("Duration adequacy").extracted_items[0].justification)
# ...

# Output processing costs (requires setting the pricing details for each LLM)
print("\nProcessing costs:")
print(llm_group.get_cost())
