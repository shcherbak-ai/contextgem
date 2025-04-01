# Example of optimizing extraction for cost

import os

from contextgem import DocumentLLM, LLMPricing

llm = DocumentLLM(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
    pricing_details=LLMPricing(
        input_per_1m_tokens=0.150,
        output_per_1m_tokens=0.600,
    ),  # add pricing details to track costs
)

# ... use the LLM for extraction ...

# ... monitor usage and cost ...
usage = llm.get_usage()  # get the usage details, including tokens and calls' details.
cost = llm.get_cost()  # get the cost details, including input, output, and total costs.
print(usage)
print(cost)
