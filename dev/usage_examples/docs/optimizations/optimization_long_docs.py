# Example of configuring LLM extraction to process long documents

import os

from contextgem import Document, DocumentLLM


# Define document
long_doc = Document(
    raw_text="long_document_text",
)

# ... attach aspects/concepts to the document ...

# Define and configure LLM
llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
)

# Extract data from document with specific configuration options
long_doc = llm.extract_all(
    long_doc,
    max_paragraphs_to_analyze_per_call=50,  # limit the number of paragraphs to analyze in an individual LLM call
    max_items_per_call=2,  # limit the number of aspects/concepts to analyze in an individual LLM call
    use_concurrency=True,  # optional: enable concurrent extractions
)

# ... use the extracted data ...
