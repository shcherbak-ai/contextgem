from contextgem import (
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    ModelPricing,
)


# Create a decision model. A decision model answers closed questions on a text
# with probabilities: it extracts aspects, and Boolean, label and rating concepts.
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key="your-typesafe-api-key",  # Replace with your actual API key
    role="decide",  # A label of your choice, needed for routing in a model group
    pricing_details=ModelPricing(  # optional
        input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
    ),
)

# Create a fallback LLM in case the decision model fails. It extracts
# the same aspects and concepts with its prompts, i.e. without the probabilities.
decision_model_fallback = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="decide",  # Must have the same role as the model it is a fallback for
    is_fallback=True,
)
decision_model.fallback_models = [decision_model_fallback]

# Create an LLM for everything else. This LLM has no role, i.e. it extracts
# the aspects and concepts that have no `model_role`.
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="your-openai-api-key",  # Replace with your actual API key
)

# Create a DocumentModelGroup. The aspects and concepts with `model_role="decide"`
# are extracted by the decision model, and all the other ones by the LLM.
model_group = DocumentModelGroup(models=[llm, decision_model])
