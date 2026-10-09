from contextgem import DocumentDecisionModel, DocumentLLM


# 1. A decision model that falls back to another decision model, and then to an LLM

primary_decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key="<your-typesafe-api-key>",
)

# First fallback: a decision model of another provider
fallback_decision_model = DocumentDecisionModel(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
    is_fallback=True,  # Must have the same role as the primary model (here: no role)
)

# Second fallback: an LLM. It extracts the same aspects/concepts with its own prompts,
# so the items it extracts have no probabilities (`item.probabilities` is None).
fallback_llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
    is_fallback=True,  # Must have the same role as the primary model (here: no role)
)

# Assign the fallback models to the primary one, in the order they should be tried
primary_decision_model.fallback_models = [fallback_decision_model, fallback_llm]

# Then use the primary model as usual
# document = primary_decision_model.extract_all(document)


# 2. An LLM that falls back to a decision model

primary_llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
)

llm_fallback_decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key="<your-typesafe-api-key>",
    is_fallback=True,  # Must have the same role as the primary model (here: no role)
)

primary_llm.fallback_models = [llm_fallback_decision_model]

# An extraction method sends requests only if the decision model can extract everything
# that the method gives to the LLM, i.e. aspects, and Boolean, label and rating concepts,
# with the settings that a decision model supports. Otherwise, it raises `ValueError`
# before any request is sent.
# document = primary_llm.extract_all(document)
