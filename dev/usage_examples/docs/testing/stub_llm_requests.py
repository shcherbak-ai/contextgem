# Testing your own code without sending requests to the LLM: the stand-in
# replaces litellm.acompletion, which ContextGem calls for each request
# to an LLM

from unittest.mock import patch

import litellm

from contextgem import Document, DocumentLLM, StringConcept


# The code to test: gets the number of an invoice from its text
def get_invoice_number(text: str, llm: DocumentLLM) -> str | None:
    document = Document(raw_text=text)
    document.add_concepts(
        [
            StringConcept(
                name="Invoice number",
                description="The number of the invoice",
                singular_occurrence=True,
            )
        ]
    )
    llm.extract_concepts_from_document(document)
    items = document.concepts[0].extracted_items
    return items[0].value if items else None


# The answer of the LLM, in the format that the extraction prompt asks for:
# a JSON array with an entry for each concept, by its ID in the prompt ("C0")
answer = '[{"concept_id": "C0", "extracted_items": ["INV-001"]}]'

# The requests that ContextGem sends, e.g. to check their prompts
requests = []
real_acompletion = litellm.acompletion


async def fake_acompletion(**kwargs):
    requests.append(kwargs)
    # With `mock_response`, litellm returns a response with this answer
    # without sending the request
    return await real_acompletion(**kwargs, mock_response=answer)


# No API key, neither here nor in the environment of the tests: a request that
# is not stubbed then fails instead of being billed
llm = DocumentLLM(model="openai/gpt-6-luna")

with patch.object(litellm, "acompletion", fake_acompletion):
    invoice_number = get_invoice_number(
        "Invoice INV-001 is due on 30 November 2026.", llm
    )

assert invoice_number == "INV-001"
assert len(requests) == 1
assert "Invoice INV-001" in requests[0]["messages"][-1]["content"]
