from contextgem import DocumentLLM, DocumentLLMGroup


# Create LLMs with different roles

# LLM without a role: extracts all aspects/concepts that are not routed to another LLM
extractor = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
    output_language="adapt",
)

# LLM with a role: extracts the aspects/concepts that have `llm_role="reasoner"`
reasoner = DocumentLLM(
    model="openai/gpt-6.1-sol",
    api_key="<your-openai-api-key>",
    role="reasoner",  # a label of your choice
    max_completion_tokens=16000,
    reasoning_effort="high",
    output_language="adapt",
)

# Create a group
llm_group = DocumentLLMGroup(
    llms=[extractor, reasoner],
    output_language="adapt",  # All LLMs in the group must share the same output language setting
)

# Then use the group as usual
# document = llm_group.extract_all(document)
