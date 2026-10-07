from contextgem import DocumentLLM


llm = DocumentLLM(
    model="anthropic/claude-haiku-4-5",
    api_key="<your-anthropic-api-key>",
    auto_pricing=True,  # or set `pricing_details=LLMPricing(...)` manually
)

# Perform some extraction tasks

# Get usage statistics
usage_info = llm.get_usage()
usage = usage_info[0].usage  # usage of the primary LLM
print(f"Input tokens: {usage.input}")
print(f"Output tokens: {usage.output}")
# Input tokens read from / written to the provider's prompt cache (part of `input`)
print(f"Cache read tokens: {usage.cache_read}")
print(f"Cache write tokens: {usage.cache_write}")

# Get cost statistics
cost_info = llm.get_cost()
cost = cost_info[0].cost  # cost of the primary LLM
print(f"Input cost: {cost.input}")  # includes the cost of the cache tokens
print(f"Output cost: {cost.output}")
print(f"Total cost: {cost.total}")

# Reset usage and cost statistics
llm.reset_usage_and_cost()

# The same methods are available for LLM groups, with optional filtering by LLM role
# usage_info = llm_group.get_usage(llm_role="reasoner")
# cost_info = llm_group.get_cost(llm_role="reasoner")
# llm_group.reset_usage_and_cost(llm_role="reasoner")
