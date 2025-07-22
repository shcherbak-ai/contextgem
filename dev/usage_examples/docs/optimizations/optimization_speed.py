# Example of optimizing extraction for speed

import os

from aiolimiter import AsyncLimiter

from contextgem import Document, DocumentLLM


# Define document
document = Document(
    raw_text="document_text",
    # aspects=[Aspect(...), ...],
    # concepts=[Concept(...), ...],
)

# Define LLM with a fallback model
llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    async_limiter=AsyncLimiter(
        10, 5
    ),  # e.g. 10 acquisitions per 5-second period; adjust to your LLM API setup
    fallback_llm=DocumentLLM(
        model="openai/gpt-3.5-turbo",
        api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
        is_fallback=True,
        async_limiter=AsyncLimiter(
            20, 5
        ),  # e.g. 20 acquisitions per 5-second period; adjust to your LLM API setup
    ),
)

# Use the LLM for extraction with concurrency enabled
llm.extract_all(document, use_concurrency=True)

# ... use the extracted data ...
