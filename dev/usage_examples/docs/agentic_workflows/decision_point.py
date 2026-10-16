# ContextGem: Deciding the Next Step of a Workflow with a Decision Model

import os

from contextgem import BooleanConcept, Document, DocumentDecisionModel


decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
)

# The document that the next step depends on
document = Document(
    raw_text=(
        "Service Agreement\n"
        "This agreement is made on 1 March 2026 between Acme Ltd (Customer) "
        "and Northwind AS (Supplier).\n"
        "The Supplier shall provide cloud hosting services as described "
        "in Schedule 1.\n"
        "This agreement renews automatically for successive one-year terms.\n"
        "Either party may terminate this agreement for any reason by giving "
        "30 days' written notice to the other party.\n"
        "This agreement is governed by the laws of Norway.\n"
    ),
)
question = BooleanConcept(
    name="Termination for convenience",
    description="Either party may terminate the agreement without cause",
)
document.add_concepts([question])
document = decision_model.extract_all(document)

concept = document.get_concept_by_name("Termination for convenience")
p_yes = concept.extracted_items[0].probabilities[True]  # 0.98

# Act on a clear answer, and leave an unclear one to a person
if p_yes >= 0.9:
    next_step = "draft_termination_notice"  # the step taken here
elif p_yes <= 0.1:
    next_step = "reply_not_terminable"
else:
    next_step = "human_review"
