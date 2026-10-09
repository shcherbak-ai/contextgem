# ContextGem: Usage and Cost of a Decision Model

import os

from contextgem import BooleanConcept, Document, DocumentDecisionModel, ModelPricing


# Configure the decision model with the prices for cost calculation
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("TYPESAFE_API_KEY"),
    pricing_details=ModelPricing(
        input_per_1m_tokens=0.042,  # Cost per 1M input tokens
        output_per_1m_tokens=0.0,  # No charge for output tokens
    ),
    # or set `auto_pricing=True` instead of `pricing_details`, where supported
)

# Perform an extraction task
doc = Document(raw_text="This Agreement is governed by the laws of Norway.")
doc.add_concepts(
    [
        BooleanConcept(
            name="Governing law",
            description="The document states which law governs the agreement",
        )
    ]
)
decision_model.extract_concepts_from_document(doc)

# Get usage statistics
usage_info = decision_model.get_usage()
usage = usage_info[0].usage  # usage of the primary model
print(f"Input tokens: {usage.input}")
print(f"Output tokens: {usage.output}")
print(f"Requests: {len(usage.calls)}")

# Each request is recorded as a call
for call in usage.calls:
    print(f"Request: {call.prompt}")  # the model, the text and the questions, as JSON
    print(f"Answers: {call.response}")  # the answers of the model, as JSON

# Get cost statistics
cost_info = decision_model.get_cost()
cost = cost_info[0].cost  # cost of the primary model
print(f"Input cost: {cost.input}")
print(f"Output cost: {cost.output}")
print(f"Total cost: {cost.total}")

# Reset usage and cost statistics
decision_model.reset_usage_and_cost()
