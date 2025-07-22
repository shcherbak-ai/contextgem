# Example of selecting different LLMs for different tasks

import os

from contextgem import Aspect, Document, DocumentLLM, DocumentLLMGroup, StringConcept


# Define LLMs
base_llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    role="extractor_text",  # default
)

# Optional - attach a fallback LLM
base_llm_fallback = DocumentLLM(
    model="openai/gpt-3-5-turbo",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    role="extractor_text",  # must have the same role as the parent LLM
    is_fallback=True,
)
base_llm.fallback_llm = base_llm_fallback

advanced_llm = DocumentLLM(
    model="openai/gpt-4o",  # can be a larger model (reasoning or non-reasoning)
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    role="reasoner_text",
)

# You can organize LLMs in a group to use them in a pipeline
llm_group = DocumentLLMGroup(
    llms=[base_llm, advanced_llm],
)

# Assign the existing LLMs to aspects/concepts
document = Document(
    raw_text="document_text",
    aspects=[
        Aspect(
            name="aspect_name",
            description="aspect_description",
            llm_role="extractor_text",
            concepts=[
                StringConcept(
                    name="concept_name",
                    description="concept_description",
                    llm_role="reasoner_text",
                )
            ],
        )
    ],
)

# Then use the LLM group to extract all information from the document
# This will use different LLMs for different aspects/concepts under the hood
# document = llm_group.extract_all(document)
