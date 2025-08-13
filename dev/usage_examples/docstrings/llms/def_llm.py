from contextgem import DocumentLLM, LLMPricing


# Create a single LLM for text extraction
text_extractor = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="your-api-key",  # Replace with your actual API key
    role="extractor_text",  # Role for text extraction
    pricing_details=LLMPricing(  # optional
        input_per_1m_tokens=0.150, output_per_1m_tokens=0.600
    ),
    # or set `auto_pricing=True` to automatically fetch pricing data from the LLM provider
)

# Create a fallback LLM in case the primary model fails
fallback_text_extractor = DocumentLLM(
    model="anthropic/claude-3-7-sonnet",
    api_key="your-anthropic-api-key",  # Replace with your actual API key
    role="extractor_text",  # must be the same as the role of the primary LLM
    is_fallback=True,
    pricing_details=LLMPricing(  # optional
        input_per_1m_tokens=3.00, output_per_1m_tokens=15.00
    ),
    # or set `auto_pricing=True` to automatically fetch pricing data from the LLM provider
)
# Assign the fallback LLM to the primary LLM
text_extractor.fallback_llm = fallback_text_extractor
