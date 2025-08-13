from contextgem import DocumentLLM, LLMPricing


# Option 1: Set up a LLM with pricing details
llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="<your-openai-api-key>",
    pricing_details=LLMPricing(
        input_per_1m_tokens=0.150,  # Cost per 1M input tokens
        output_per_1m_tokens=0.600,  # Cost per 1M output tokens
    ),
)

# Option 2: Set up a LLM with auto-pricing
llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="<your-openai-api-key>",
    auto_pricing=True,
)

# Perform some extraction tasks

# Later, you can check the cost
cost_info = llm.get_cost()
