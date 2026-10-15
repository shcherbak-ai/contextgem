from contextgem import ModelPricing


# Create a pricing model for an LLM, with the prices of your provider
# (example values)
pricing = ModelPricing(
    input_per_1m_tokens=1.00,  # $1.00 per million input tokens
    output_per_1m_tokens=2.00,  # $2.00 per million output tokens
    # Optional: $0.10 per million input tokens read from the provider's prompt cache
    cache_read_per_1m_tokens=0.10,
)

# ModelPricing objects are immutable
try:
    pricing.input_per_1m_tokens = 0.7
except ValueError as e:
    print(f"Error when trying to modify pricing: {e}")
