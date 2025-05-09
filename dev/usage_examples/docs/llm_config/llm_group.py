from contextgem import DocumentLLM, DocumentLLMGroup

# Create LLMs with different roles
text_extractor = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key="<your-openai-api-key>",
    role="extractor_text",
    output_language="adapt",
)

text_reasoner = DocumentLLM(
    model="openai/o3-mini",
    api_key="<your-openai-api-key>",
    role="reasoner_text",
    max_completion_tokens=16000,
    reasoning_effort="high",
    output_language="adapt",
)

# Create a group
llm_group = DocumentLLMGroup(
    llms=[text_extractor, text_reasoner],
    output_language="adapt",  # All LLMs in the group must share the same output language setting
)

# Then use the group as usual
# document = llm_group.extract_all(document)
