from contextgem import LLMPricing


# Create a pricing model for an LLM (openai/o3-mini example)
pricing = LLMPricing(
    input_per_1m_tokens=1.10,  # $1.10 per million input tokens
    output_per_1m_tokens=4.40,  # $4.40 per million output tokens
)

# LLMPricing objects are immutable
try:
    pricing.input_per_1m_tokens = 0.7
except ValueError as e:
    print(f"Error when trying to modify pricing: {e}")
