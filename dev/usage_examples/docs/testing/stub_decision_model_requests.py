# Testing your own code without sending requests to the decision model:
# the stand-in replaces litellm.adecisions, which ContextGem calls for each
# request to a decision model

from unittest.mock import patch

import litellm
from litellm.types.decisions import (
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
)

from contextgem import BooleanConcept, Document, DocumentDecisionModel


async def fake_adecisions(**kwargs):
    # One answer for each question of the request, in the order of
    # the questions: here "yes", with a probability of 0.9, to each
    # yes/no question ("predicate")
    answers = tuple(
        OpenAIPredicateAnswer(name=question["name"], probability=0.9)
        for question in kwargs["questions"]
    )
    usage = OpenAIDecisionUsage(
        input_tokens=100,
        output_tokens=0,
        total_tokens=100,
    )
    return OpenAIDecisionResponse(
        model=kwargs["model"],
        answers=answers,
        usage=usage,
    )


text = "The supplier may end the agreement on 30 days' notice."
document = Document(raw_text=text)
document.add_concepts(
    [
        BooleanConcept(
            name="Supplier termination right",
            description="Whether the supplier may terminate the agreement",
        )
    ]
)

# No API key: a request that is not stubbed fails instead of being billed
decision_model = DocumentDecisionModel(model="typesafe/jev-latest")

with patch.object(litellm, "adecisions", fake_adecisions):
    decision_model.extract_concepts_from_document(document)

item = document.concepts[0].extracted_items[0]
assert item.value is True
assert item.probabilities[True] == 0.9
