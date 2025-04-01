from contextgem import NumericalConcept

# Create concepts for different numerical values in the contract
payment_amount = NumericalConcept(
    name="Payment amount",
    description="The monetary value to be paid according to the contract terms",
    numeric_type="float",
    llm_role="extractor_text",
    add_references=True,
    reference_depth="sentences",
)

payment_days = NumericalConcept(
    name="Payment term days",
    description="The number of days within which payment must be made",
    numeric_type="int",
    llm_role="extractor_text",
    add_justifications=True,
    justification_depth="balanced",
)
