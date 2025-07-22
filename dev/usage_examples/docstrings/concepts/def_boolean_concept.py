from contextgem import BooleanConcept


# Create the concept with specific configuration
has_confidentiality = BooleanConcept(
    name="Contains confidentiality clause",
    description="Determines whether the contract includes provisions requiring parties to maintain confidentiality",
    llm_role="reasoner_text",
    singular_occurrence=True,
    add_justifications=True,
    justification_depth="brief",
)
