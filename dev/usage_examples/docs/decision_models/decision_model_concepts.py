# ContextGem: Concept Extraction with a Decision Model

import os

from contextgem import (
    BooleanConcept,
    Document,
    DocumentDecisionModel,
    LabelConcept,
    RatingConcept,
)


# Create a document instance
doc = Document(
    raw_text=(
        "Consultancy Agreement\n"
        "This agreement is made between Company A (Supplier) and Company B (Customer).\n"
        "The Supplier shall provide the consultancy services described in Annex 1.\n"
        "The Customer shall pay the fees within 30 calendar days of receiving an invoice. "
        "Late payments bear interest at 1% per month.\n"
        "Either party may terminate this agreement at any time by giving 30 days' "
        "written notice to the other party.\n"
        "Each party shall keep the terms of this agreement confidential.\n"
    ),
)

# Define the concepts. A decision model extracts the concepts whose values come
# from a closed set of options: Boolean, label and rating concepts.
doc.add_concepts(
    [
        # A yes/no question
        BooleanConcept(
            name="Termination for convenience",
            description="Either party may terminate the agreement without cause",
        ),
        # One label out of several
        LabelConcept(
            name="Contract type",
            description="Type of the contract",
            labels=["NDA", "Consultancy agreement", "Lease agreement", "Other"],
            classification_type="multi_class",
        ),
        # Any number of labels: each label is decided on its own
        LabelConcept(
            name="Topics covered",
            description="Topics that the contract has provisions on",
            labels=["Payment", "Termination", "Confidentiality", "Liability"],
            classification_type="multi_label",
        ),
        # A level of a scale
        RatingConcept(
            name="Payment terms detail",
            description=(
                "How detailed the payment terms are, from 1 (payment is not addressed) "
                "to 5 (amounts, deadlines and late payment consequences are specified)"
            ),
            rating_scale=(1, 5),
        ),
    ]
)

# Configure the decision model with your API parameters
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
)

# Extract the concepts: the questions of all the concepts are sent in one request,
# together with the text of the document
extracted_concepts = decision_model.extract_concepts_from_document(doc)
# or use async version `await decision_model.extract_concepts_from_document_async(doc)`

# Each extracted item has a value and the probabilities of all the options
for concept in extracted_concepts:
    print(concept.name)
    for item in concept.extracted_items:
        print(f"  value: {item.value}")
        print(f"  probabilities: {item.probabilities}")

# The output has the following form. The probabilities below are made up to show
# the format: they are not the output of a model.
#
# Termination for convenience
#   value: True
#   probabilities: {True: 0.96, False: 0.04}
# Contract type
#   value: ['Consultancy agreement']
#   probabilities: {'NDA': 0.02, 'Consultancy agreement': 0.95, 'Lease agreement': 0.01, 'Other': 0.02}
# Topics covered
#   value: ['Payment', 'Termination', 'Confidentiality']
#   probabilities: {'Payment': 0.98, 'Termination': 0.97, 'Confidentiality': 0.91, 'Liability': 0.06}
# Payment terms detail
#   value: 3
#   probabilities: {1: 0.03, 2: 0.21, 3: 0.48, 4: 0.24, 5: 0.04}

# The probability of the returned value is the model's own estimate of it. It can be
# used e.g. to send the less certain answers to a human review.
termination_concept = doc.get_concept_by_name("Termination for convenience")
# A Boolean concept always gets an item from a decision model
termination_item = termination_concept.extracted_items[0]
# `probabilities` is None only for an item extracted by an LLM
probabilities = termination_item.probabilities
if probabilities is not None and probabilities[termination_item.value] < 0.8:
    print("Termination for convenience: uncertain answer, to be reviewed")
