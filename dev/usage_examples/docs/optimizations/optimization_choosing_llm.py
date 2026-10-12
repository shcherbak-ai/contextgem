# Example of selecting different LLMs for different tasks

import os

from contextgem import Aspect, Document, DocumentLLM, DocumentModelGroup, StringConcept


# Define LLMs

# LLM without a role - extracts the aspects/concepts that have no `model_role`
base_llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
)

# Optional - attach a fallback LLM
base_llm_fallback = DocumentLLM(
    model="openai/gpt-5.6-luna",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    is_fallback=True,  # must have the same role as the primary LLM (here: no role)
)
base_llm.fallback_models = [base_llm_fallback]

# LLM with a role - extracts the aspects/concepts with the same `model_role`
advanced_llm = DocumentLLM(
    model="openai/gpt-6.1-sol",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    role="reasoner",  # a label of your choice
)

# You can organize LLMs in a group to use them in a pipeline
model_group = DocumentModelGroup(
    models=[base_llm, advanced_llm],
)

# Route aspects/concepts to the LLMs in the group
document = Document(
    raw_text="document_text",
    aspects=[
        Aspect(
            name="aspect_name",
            description="aspect_description",
            # no `model_role` - extracted by the LLM without a role
            concepts=[
                StringConcept(
                    name="concept_name",
                    description="concept_description",
                    model_role="reasoner",  # extracted by the LLM with the "reasoner" role
                )
            ],
        )
    ],
)

# Then use the model group to extract all information from the document
# This will use different LLMs for different aspects/concepts under the hood
# document = model_group.extract_all(document)
