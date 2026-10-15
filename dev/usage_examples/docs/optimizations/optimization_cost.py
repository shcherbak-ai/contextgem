# Example of optimizing extraction for cost

import os

from contextgem import DocumentLLM


llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    auto_pricing=True,  # calculate the costs with `genai-prices`, to track them
    # or set `pricing_details=ModelPricing(...)` with the prices of your provider
)

# ... use the LLM for extraction ...

# ... monitor usage and cost ...
usage = llm.get_usage()  # get the usage details, including tokens and calls' details.
cost = llm.get_cost()  # get the cost details, including input, output, and total costs.
print(usage)
print(cost)
