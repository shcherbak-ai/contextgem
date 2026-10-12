from contextgem import DocumentLLM, ModelPricing


# Create a single LLM, which extracts all aspects and concepts it is given
# (a role is needed only for routing in a model group)
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="your-api-key",  # Replace with your actual API key
    pricing_details=ModelPricing(  # optional
        input_per_1m_tokens=0.10, output_per_1m_tokens=0.50
    ),
    # or set `auto_pricing=True` to calculate the costs automatically with `genai-prices`
)

# Create a fallback LLM in case the primary model fails
fallback_llm = DocumentLLM(
    model="anthropic/claude-sonnet-5-5",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    is_fallback=True,  # must have the same role as the primary LLM (here: no role)
    pricing_details=ModelPricing(  # optional
        input_per_1m_tokens=2.00, output_per_1m_tokens=10.00
    ),
    # or set `auto_pricing=True` to calculate the costs automatically with `genai-prices`
)
# Assign the fallback LLM to the primary LLM
# (several fallback models can be assigned, to be tried in the given order)
llm.fallback_models = [fallback_llm]
