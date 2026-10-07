from contextgem import DocumentLLM


# Primary LLM
primary_llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
)

# First fallback LLM
fallback_llm_1 = DocumentLLM(
    model="anthropic/claude-haiku-4-5",
    api_key="<your-anthropic-api-key>",
    is_fallback=True,  # Must have the same role as the primary LLM (here: no role)
)

# Second fallback LLM (optional)
fallback_llm_2 = DocumentLLM(
    model="gemini/gemini-3.8-flash",
    api_key="<your-gemini-api-key>",
    is_fallback=True,  # Must have the same role as the primary LLM (here: no role)
)

# Assign fallback LLMs to primary, in the order they should be tried
primary_llm.fallback_llms = [fallback_llm_1, fallback_llm_2]

# Then use the primary LLM as usual
# document = primary_llm.extract_all(document)
