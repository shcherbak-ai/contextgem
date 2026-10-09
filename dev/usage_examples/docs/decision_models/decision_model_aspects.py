# ContextGem: Aspect Extraction with a Decision Model

import os

from contextgem import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentDecisionModel,
    LabelConcept,
)


# Create a document instance. Each line of the text is a paragraph.
doc = Document(
    raw_text=(
        "Service Agreement\n"
        "The Client shall pay the Provider a monthly service fee of $5,000.\n"
        "Late payments will incur a penalty of 2% per month on the outstanding balance.\n"
        "Either party may terminate this Agreement for any reason by giving 30 days' "
        "written notice to the other party.\n"
        "The Provider may terminate this Agreement immediately if a payment is more "
        "than 60 days overdue.\n"
        "This Agreement is governed by the laws of the State of New York.\n"
    ),
)

# Define the aspects, each with its own concepts
doc.add_aspects(
    [
        Aspect(
            name="Payment terms",
            description="Clauses on fees, payment deadlines and late payments",
            concepts=[
                BooleanConcept(
                    name="Late payment penalty",
                    description="A penalty or interest is charged on late payments",
                ),
            ],
        ),
        Aspect(
            name="Termination",
            description="Clauses on termination of the agreement",
            concepts=[
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
            ],
        ),
    ]
)

# Configure the decision model with your API parameters
decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("TYPESAFE_API_KEY"),
)

# Extract the aspects and their concepts.
# - Aspects: each paragraph of the document is sent in a request of its own, with
#   one yes/no question per aspect. Here: 6 requests, each with 2 questions.
# - Concepts: the questions of the concepts of an aspect are sent in one request,
#   with the text of the paragraphs extracted for that aspect.
doc = decision_model.extract_all(
    doc,
    # Send the requests of the paragraphs concurrently, at the rate allowed
    # by the `async_limiter` of the model
    use_concurrency=True,
    # With concurrency, each aspect and each concept is extracted in a call of its
    # own by default, i.e. each text is sent once per aspect or concept. Allow for
    # up to 10 aspects (or concepts) per call, to send each text once.
    max_items_per_call=10,
)
# or use async version `await decision_model.extract_all_async(doc, ...)`

for aspect in doc.aspects:
    print(aspect.name)
    # Each extracted item of an aspect is a paragraph of the document
    for item in aspect.extracted_items:
        # Probability that the paragraph belongs to the aspect
        # (`probabilities` is None only for an item extracted by an LLM)
        if item.probabilities is not None:
            print(f"  [{item.probabilities[True]:.2f}] {item.value}")
    # Concepts extracted from the paragraphs of the aspect
    for concept in aspect.concepts:
        for item in concept.extracted_items:
            print(f"  {concept.name}: {item.value} {item.probabilities}")
