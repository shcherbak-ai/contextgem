from contextgem import DocumentLLM


# Primary LLM
primary_llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="<your-openai-api-key>",
    role="extractor_text",  # default role
)

# Fallback LLM
fallback_llm = DocumentLLM(
    model="anthropic/claude-3-5-haiku",
    api_key="<your-anthropic-api-key>",
    role="extractor_text",  # Must match the primary LLM's role
    is_fallback=True,
)

# Assign fallback LLM to primary
primary_llm.fallback_llm = fallback_llm

# Then use the primary LLM as usual
# document = primary_llm.extract_all(document)
