from contextgem import DocumentLLM, DocumentLLMGroup


# Create an extractor LLM with a fallback. This LLM has no role, i.e. it extracts
# all aspects and concepts that are not routed to another LLM in the group.
extractor = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="your-openai-api-key",  # Replace with your actual API key
)

# Create a fallback LLM for the extractor
extractor_fallback = DocumentLLM(
    model="anthropic/claude-3-5-haiku",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    is_fallback=True,  # Must have the same role as the primary LLM (here: no role)
)

# Assign the fallback LLM to the primary extractor
extractor.fallback_llms = [extractor_fallback]

# Create a reasoner LLM. A role is a label of your choice: this LLM extracts
# the aspects and concepts that have `llm_role="reasoner"`.
reasoner = DocumentLLM(
    model="openai/o3-mini",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="reasoner",  # For more complex tasks that require reasoning
)

# Create an LLM dedicated to images. This LLM extracts the concepts that have
# `llm_role="vision"`, e.g. the concepts with `modality="image"`.
vision_extractor = DocumentLLM(
    model="openai/gpt-5-mini",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="vision",  # For extraction from images
)

# Create a DocumentLLMGroup with all three LLMs
llm_group = DocumentLLMGroup(
    llms=[extractor, reasoner, vision_extractor],
    output_language="en",  # All LLMs must have the same output language ("en" is default)
)
# This group will have 4 LLMs: three main ones, with different roles,
# and one fallback LLM for a specific LLM. Each LLM can have one or more fallback LLMs.

# Get usage statistics for the whole group or for a specific role
group_usage = llm_group.get_usage()
reasoner_usage = llm_group.get_usage(llm_role="reasoner")

# Get cost statistics for the whole group or for a specific role
all_costs = llm_group.get_cost()
reasoner_cost = llm_group.get_cost(llm_role="reasoner")

# Reset usage and cost statistics for the whole group or for a specific role
llm_group.reset_usage_and_cost()
llm_group.reset_usage_and_cost(llm_role="reasoner")
