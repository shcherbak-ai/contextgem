# Example of optimizing extraction for cost

import os

from contextgem import DocumentLLM, LLMPricing


llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    pricing_details=LLMPricing(
        input_per_1m_tokens=0.10,
        output_per_1m_tokens=0.50,
    ),  # add pricing details to track costs
    # or set `auto_pricing=True` to automatically fetch pricing data from the LLM provider
)

# ... use the LLM for extraction ...

# ... monitor usage and cost ...
usage = llm.get_usage()  # get the usage details, including tokens and calls' details.
cost = llm.get_cost()  # get the cost details, including input, output, and total costs.
print(usage)
print(cost)
