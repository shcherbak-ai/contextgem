from contextgem import DateConcept


# Create a date concept to extract the effective date of the contract
effective_date = DateConcept(
    name="Effective date",
    description="The effective date as specified in the contract",
    add_references=True,  # Include references to where dates were found
    singular_occurrence=True,  # Only extract one effective date per document
)
