# ContextGem: Usage and Cost of a Decision Model

import os

from contextgem import BooleanConcept, Document, DocumentDecisionModel


# Configure the decision model with automatic pricing for cost calculation
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
    # The prices of the decision requests of the TypeSafe models are in
    # `genai-prices`. For a model whose prices it does not have, e.g. a model
    # deployed in Microsoft Foundry, set `pricing_details=ModelPricing(...)`
    # with the prices of your provider instead
    auto_pricing=True,
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
