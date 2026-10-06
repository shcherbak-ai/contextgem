# Example of selecting different LLMs for different tasks

import os

from contextgem import Aspect, Document, DocumentLLM, DocumentLLMGroup, StringConcept


# Define LLMs

# LLM without a role - extracts all aspects/concepts that are not routed to another LLM
base_llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
)

# Optional - attach a fallback LLM
base_llm_fallback = DocumentLLM(
    model="openai/gpt-3-5-turbo",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    is_fallback=True,  # must have the same role as the parent LLM (here: no role)
)
base_llm.fallback_llms = [base_llm_fallback]

# LLM with a role - extracts the aspects/concepts with the same `llm_role`
advanced_llm = DocumentLLM(
    model="openai/o3-mini",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    role="reasoner",  # a label of your choice
)

# You can organize LLMs in a group to use them in a pipeline
llm_group = DocumentLLMGroup(
    llms=[base_llm, advanced_llm],
)

# Route aspects/concepts to the LLMs in the group
document = Document(
    raw_text="document_text",
    aspects=[
        Aspect(
            name="aspect_name",
            description="aspect_description",
            # no `llm_role` - extracted by the LLM without a role
            concepts=[
                StringConcept(
                    name="concept_name",
                    description="concept_description",
                    llm_role="reasoner",  # extracted by the LLM with the "reasoner" role
                )
            ],
        )
    ],
)

# Then use the LLM group to extract all information from the document
# This will use different LLMs for different aspects/concepts under the hood
# document = llm_group.extract_all(document)
