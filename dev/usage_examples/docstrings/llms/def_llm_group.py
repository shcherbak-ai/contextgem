from contextgem import DocumentLLM, DocumentLLMGroup

# Create a text extractor LLM with a fallback
text_extractor = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="extractor_text",
)

# Create a fallback LLM for the text extractor
text_extractor_fallback = DocumentLLM(
    model="anthropic/claude-3-5-haiku",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    role="extractor_text",  # Must have the same role as the primary LLM
    is_fallback=True,
)

# Assign the fallback LLM to the primary text extractor
text_extractor.fallback_llm = text_extractor_fallback

# Create a text reasoner LLM
text_reasoner = DocumentLLM(
    model="openai/o3-mini",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="reasoner_text",  # For more complex tasks that require reasoning
)

# Create a vision extractor LLM
vision_extractor = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="your-openai-api-key",  # Replace with your actual API key
    role="extractor_vision",  # For handling images
)

# Create a vision reasoner LLM
vision_reasoner = DocumentLLM(
    model="openai/gpt-4o",
    api_key="your-openai-api-key",
    role="reasoner_vision",  # For more complex vision tasks that require reasoning
)

# Create a DocumentLLMGroup with all four LLMs
llm_group = DocumentLLMGroup(
    llms=[text_extractor, text_reasoner, vision_extractor, vision_reasoner],
    output_language="en",  # All LLMs must have the same output language ("en" is default)
)
# This group will have 5 LLMs: four main ones, with different roles,
# and one fallback LLM for a specific LLM. Each LLM can have a fallback LLM.

# Get usage statistics for the whole group or for a specific role
group_usage = llm_group.get_usage()
text_extractor_usage = llm_group.get_usage(llm_role="extractor_text")

# Get cost statistics for the whole group or for a specific role
all_costs = llm_group.get_cost()
text_extractor_cost = llm_group.get_cost(llm_role="extractor_text")

# Reset usage and cost statistics for the whole group or for a specific role
llm_group.reset_usage_and_cost()
llm_group.reset_usage_and_cost(llm_role="extractor_text")
