from contextgem import DocumentLLM


# Primary LLM
primary_llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="<your-openai-api-key>",
    role="extractor_text",  # default role
)

# First fallback LLM
fallback_llm_1 = DocumentLLM(
    model="anthropic/claude-3-5-haiku",
    api_key="<your-anthropic-api-key>",
    role="extractor_text",  # Must match the primary LLM's role
    is_fallback=True,
)

# Second fallback LLM (optional)
fallback_llm_2 = DocumentLLM(
    model="gemini/gemini-2.0-flash",
    api_key="<your-gemini-api-key>",
    role="extractor_text",  # Must match the primary LLM's role
    is_fallback=True,
)

# Assign fallback LLMs to primary, in the order they should be tried
primary_llm.fallback_llms = [fallback_llm_1, fallback_llm_2]

# Then use the primary LLM as usual
# document = primary_llm.extract_all(document)
