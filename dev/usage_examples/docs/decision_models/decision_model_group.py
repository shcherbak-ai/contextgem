# ContextGem: Using a Decision Model and an LLM in a Model Group

import os

from contextgem import (
    BooleanConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    LabelConcept,
    StringConcept,
)


# LLM without a role: extracts all aspects/concepts that are not routed to another model
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
)

# Decision model with a role: extracts the aspects/concepts that have `model_role="decide"`
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("TYPESAFE_API_KEY"),
    role="decide",  # a label of your choice
)

# Create a group
model_group = DocumentModelGroup(models=[llm, decision_model])

# Create a document instance
doc = Document(
    raw_text=(
        "Service Agreement\n"
        "Either party may terminate this Agreement for any reason by giving 30 days' "
        "written notice to the other party.\n"
        "This Agreement is governed by the laws of the State of New York.\n"
    ),
)

# Route the concepts to the models in the group
doc.add_concepts(
    [
        # Closed questions: extracted by the decision model
        BooleanConcept(
            name="Termination for convenience",
            description="Either party may terminate the agreement without cause",
            model_role="decide",
        ),
        LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=["Written", "Email", "Any form", "Not specified"],
            classification_type="multi_class",
            model_role="decide",
        ),
        # Open value: no `model_role`, i.e. extracted by the LLM
        StringConcept(
            name="Notice period",
            description="Length of the termination notice period",
        ),
    ]
)

# Use the group as a single model
doc = model_group.extract_all(doc)
# or use async version `await model_group.extract_all_async(doc)`

for concept in doc.concepts:
    for item in concept.extracted_items:
        # `probabilities` is None for the items extracted by the LLM
        print(f"{concept.name}: {item.value} {item.probabilities}")

# Usage of the decision model only
decision_model_usage = model_group.get_usage(model_role="decide")
print(f"Requests to the decision model: {len(decision_model_usage[0].usage.calls)}")
