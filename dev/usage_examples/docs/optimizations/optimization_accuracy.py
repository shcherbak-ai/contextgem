# Example of optimizing extraction for accuracy

import os

from contextgem import Document, DocumentLLM, StringConcept, StringExample

# Define document
doc = Document(
    raw_text="Non-Disclosure Agreement...",
    sat_model_id="sat-6l-sm",  # default is "sat-3l-sm"
    paragraph_segmentation_mode="sat",  # default is "newlines"
    # sentence segmentation mode is always "sat", as other approaches proved to be less accurate
)

# Define document concepts
doc.concepts = [
    StringConcept(
        name="Title",  # A very simple concept, just an example for testing purposes
        description="Title of the document",
        add_justifications=True,  # enable justifications
        justification_depth="brief",  # default
        examples=[
            StringExample(
                content="Supplier Agreement",
            )
        ],
    ),
    # ... add other concepts ...
]

# ... attach other aspects/concepts to the document ...

# Define and configure LLM
llm = DocumentLLM(
    model="openai/gpt-4o",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    fallback_llm=DocumentLLM(
        model="openai/gpt-4-turbo",
        api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
        is_fallback=True,
    ),  # configure a fallback LLM
)

# Extract data from document with specific configuration options
doc = llm.extract_all(
    doc,
    max_paragraphs_to_analyze_per_call=30,  # limit the number of paragraphs to analyze in an individual LLM call
    max_items_per_call=1,  # limit the number of aspects/concepts to analyze in an individual LLM call
    use_concurrency=True,  # optional: enable concurrent extractions
)

# ... use the extracted data ...
