from contextgem import DocumentLLM, DocumentModelGroup


# Create an extractor LLM with a fallback. This LLM has no role, i.e. it extracts
# the aspects and concepts that have no `model_role`.
extractor = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="your-openai-api-key",  # Replace with your actual API key
)

# Create a fallback LLM for the extractor
extractor_fallback = DocumentLLM(
    model="anthropic/claude-haiku-4-5",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    is_fallback=True,  # Must have the same role as the primary LLM (here: no role)
)

# Assign the fallback LLM to the primary extractor
extractor.fallback_models = [extractor_fallback]

# Create a reasoner LLM. A role is a label of your choice: this LLM extracts
# the aspects and concepts that have `model_role="reasoner"`.
reasoner = DocumentLLM(
    model="openai/gpt-6.1-sol",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="reasoner",  # For more complex tasks that require reasoning
)

# Create an LLM dedicated to images. This LLM extracts the concepts that have
# `model_role="vision"`, e.g. the concepts with `modality="image"`.
vision_extractor = DocumentLLM(
    model="openai/gpt-5.6-luna",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="vision",  # For extraction from images
)

# Create a DocumentModelGroup with all three LLMs
model_group = DocumentModelGroup(
    models=[extractor, reasoner, vision_extractor],
    output_language="en",  # All LLMs must have the same output language ("en" is default)
)
# This group will have 4 LLMs: three main ones (one without a role, two with roles),
# and one fallback LLM for a specific LLM. Each LLM can have one or more fallback models.

# Get usage statistics for the whole group or for a specific role
group_usage = model_group.get_usage()
reasoner_usage = model_group.get_usage(model_role="reasoner")

# Get cost statistics for the whole group or for a specific role
all_costs = model_group.get_cost()
reasoner_cost = model_group.get_cost(model_role="reasoner")

# Reset usage and cost statistics for the whole group or for a specific role
model_group.reset_usage_and_cost()
model_group.reset_usage_and_cost(model_role="reasoner")
