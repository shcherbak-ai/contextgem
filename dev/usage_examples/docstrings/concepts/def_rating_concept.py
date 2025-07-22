from contextgem import RatingConcept


# Create a concept to rate the fairness of contract terms
fairness_rating = RatingConcept(
    name="Contract fairness rating",
    description="Evaluation of how balanced and fair the contract terms are for all parties",
    rating_scale=(1, 5),
    llm_role="reasoner_text",
    add_justifications=True,
    justification_depth="comprehensive",
    justification_max_sents=10,
)

# Create a concept to rate the clarity of contract language
clarity_rating = RatingConcept(
    name="Language clarity rating",
    description="Assessment of how clear and unambiguous the contract language is",
    rating_scale=(1, 10),
    llm_role="reasoner_text",
    add_justifications=True,
    justification_depth="balanced",
    justification_max_sents=3,
)
