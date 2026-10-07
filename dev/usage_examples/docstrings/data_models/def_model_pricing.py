from contextgem import ModelPricing


# Create a pricing model for an LLM (openai/gpt-6-luna example)
pricing = ModelPricing(
    input_per_1m_tokens=0.10,  # $0.10 per million input tokens
    output_per_1m_tokens=0.50,  # $0.50 per million output tokens
    # Optional: $0.01 per million input tokens read from the provider's prompt cache
    cache_read_per_1m_tokens=0.01,
)

# ModelPricing objects are immutable
try:
    pricing.input_per_1m_tokens = 0.7
except ValueError as e:
    print(f"Error when trying to modify pricing: {e}")
