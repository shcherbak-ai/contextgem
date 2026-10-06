from contextgem import DocumentLLM, LLMPricing


# Create a single LLM, which extracts all aspects and concepts it is given
# (a role is needed only for routing in an LLM group)
text_extractor = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="your-api-key",  # Replace with your actual API key
    pricing_details=LLMPricing(  # optional
        input_per_1m_tokens=0.150, output_per_1m_tokens=0.600
    ),
    # or set `auto_pricing=True` to automatically fetch pricing data from the LLM provider
)

# Create a fallback LLM in case the primary model fails
fallback_text_extractor = DocumentLLM(
    model="anthropic/claude-3-7-sonnet",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    is_fallback=True,  # must have the same role as the primary LLM (here: no role)
    pricing_details=LLMPricing(  # optional
        input_per_1m_tokens=3.00, output_per_1m_tokens=15.00
    ),
    # or set `auto_pricing=True` to automatically fetch pricing data from the LLM provider
)
# Assign the fallback LLM to the primary LLM
# (several fallback LLMs can be assigned, to be tried in the given order)
text_extractor.fallback_llms = [fallback_text_extractor]
