from contextgem import Aspect, BooleanConcept, DocumentDecisionModel, LabelConcept


# The threshold of the decision model applies to each aspect and concept that
# has no threshold of its own. It is 0.5 unless set.
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key="<your-typesafe-api-key>",
    decision_threshold=0.6,
)

# A lower threshold for an aspect: also the paragraphs that the model
# is less sure of are extracted for it
liability_aspect = Aspect(
    name="Liability",
    description="Clauses on the liability of the parties",
    decision_threshold=0.3,
)

# A higher threshold for a concept: the value is True only with
# a probability of 0.9 or more
arbitration_concept = BooleanConcept(
    name="Arbitration",
    description="Disputes are settled by arbitration",
    decision_threshold=0.9,
)

# A threshold for the labels of a multi-label concept: a label is selected
# with a probability of 0.7 or more
topics_concept = LabelConcept(
    name="Topics",
    description="Topics that the document covers",
    labels=["confidentiality", "payment", "termination"],
    classification_type="multi_label",
    decision_threshold=0.7,
)

# No threshold of its own: the threshold of the decision model (0.6) applies
payment_aspect = Aspect(
    name="Payment",
    description="Clauses on the fees and on how the fees are paid",
)
