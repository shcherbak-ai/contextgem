import os

from contextgem import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentDecisionModel,
    LabelConcept,
)


doc = Document(
    raw_text=(
        "Service Agreement\n"
        "Either party may terminate this Agreement for any reason by giving 30 days' "
        "written notice to the other party.\n"
        "This Agreement is governed by the laws of the State of New York.\n"
    ),
)

doc.add_aspects(
    [
        Aspect(
            name="Termination",
            description="Clauses on termination of the agreement",
        )
    ]
)
doc.add_concepts(
    [
        BooleanConcept(
            name="Termination for convenience",
            description="Either party may terminate the agreement without cause",
        ),
        LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=["Written", "Email", "Any form", "Not specified"],
            classification_type="multi_class",
        ),
    ]
)

decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",  # or another decision model, cloud or local
    api_key=os.environ.get("CONTEXTGEM_TYPESAFE_API_KEY"),
)
# The aspect: a yes/no question for each paragraph. The concepts: both in one request.
doc = decision_model.extract_all(doc)

termination_clauses = doc.get_aspect_by_name("Termination").extracted_items
# 1 item, the paragraph: "Either party may terminate this Agreement for any reason..."
# .probabilities: {True: 0.99, False: 0.01} (rounded); the other two paragraphs
#     had 0.26 and 0.03, below the decision threshold of 0.5

convenience = doc.get_concept_by_name("Termination for convenience").extracted_items[0]
# .value: True
# .probabilities: {True: 0.98, False: 0.02} (rounded)

notice_form = doc.get_concept_by_name("Notice form").extracted_items[0]
# .value: ["Written"]
# .probabilities: {"Written": 1.0, "Email": 0.0, "Any form": 0.0, "Not specified": 0.0}
