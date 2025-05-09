from contextgem import DocumentLLM

llm = DocumentLLM(
    model="anthropic/claude-3-5-haiku",
    api_key="<your-anthropic-api-key>",
)

# Perform some extraction tasks

# Get usage statistics
usage_info = llm.get_usage()

# Get cost statistics
cost_info = llm.get_cost()

# Reset usage and cost statistics
llm.reset_usage_and_cost()

# The same methods are available for LLM groups, with optional filtering by LLM role
# usage_info = llm_group.get_usage(llm_role="extractor_text")
# cost_info = llm_group.get_cost(llm_role="extractor_text")
# llm_group.reset_usage_and_cost(llm_role="extractor_text")
