from contextgem import DocumentLLM, ModelPricing


# Option 1: Set up a LLM with pricing details
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
    pricing_details=ModelPricing(  # example values: set the prices of your provider
        input_per_1m_tokens=1.00,  # Cost per 1M input tokens
        output_per_1m_tokens=2.00,  # Cost per 1M output tokens
        # Optional: cost per 1M input tokens read from the provider's prompt cache
        cache_read_per_1m_tokens=0.10,
    ),
)

# Option 2: Set up a LLM with auto-pricing
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
    auto_pricing=True,
)

# Perform some extraction tasks

# Later, you can check the cost
cost_info = llm.get_cost()
