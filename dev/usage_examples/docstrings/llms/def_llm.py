from contextgem import DocumentLLM


# Create a single LLM, which extracts all aspects and concepts it is given
# (a role is needed only for routing in a model group)
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="your-api-key",  # Replace with your actual API key
    auto_pricing=True,  # optional, to calculate the costs with `genai-prices`
)

# Create a fallback LLM in case the primary model fails
fallback_llm = DocumentLLM(
    model="anthropic/claude-sonnet-5-5",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    is_fallback=True,  # must have the same role as the primary LLM (here: no role)
    auto_pricing=True,  # optional, to calculate the costs with `genai-prices`
)
# Assign the fallback LLM to the primary LLM
# (several fallback models can be assigned, to be tried in the given order)
llm.fallback_models = [fallback_llm]
