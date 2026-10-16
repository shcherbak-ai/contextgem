# ContextGem: Extraction as a Tool of an Agent

import json
import os

from contextgem import Document, DocumentLLM, StringConcept, register_tool


# The documents that the agent can read, e.g. loaded from your store
documents = {
    "acme-service-agreement": Document(
        raw_text=(
            "Service Agreement\n"
            "This agreement is made on 1 March 2026 between Acme Ltd "
            "(Customer) and Northwind AS (Supplier).\n"
            "The Supplier shall provide cloud hosting services as described "
            "in Schedule 1.\n"
            "This agreement renews automatically for successive one-year "
            "terms.\n"
            "Either party may terminate this agreement for any reason by "
            "giving 30 days' written notice to the other party.\n"
            "This agreement is governed by the laws of Norway.\n"
        ),
    ),
}

# The LLM that extracts, which can be another model than the agent's
extractor = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
)


@register_tool
async def extract_from_document(
    document_id: str,
    name: str,
    description: str,
) -> str:
    """
    Extracts information from a document, with the sentences it was found in.

    :param document_id: The ID of the document
    :param name: A short name of the information, e.g. "Payment term"
    :param description: What to extract, e.g. "Days to pay an invoice"
    """
    # A copy: the stored document stays as it is
    document = documents[document_id].clone()
    document.add_concepts(
        [
            StringConcept(
                name=name,
                description=description,
                add_references=True,
                reference_depth="sentences",
            )
        ]
    )
    await extractor.extract_concepts_from_document_async(document)
    items = document.get_concept_by_name(name).extracted_items
    # The result: the values and their sources, not the whole document
    return json.dumps(
        [
            {
                "value": item.value,
                "sources": [s.raw_text for s in item.reference_sentences],
            }
            for item in items
        ]
    )


# The agent: here, ContextGem's own chat with tools
agent = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
    system_message=(
        "You answer questions about the user's documents, which you read with "
        "the tool. Quote the sentences that your answer relies on. Documents: "
        "acme-service-agreement (service agreement with Acme Ltd)."
    ),
    tools=[extract_from_document],
)
question = "How much notice do we need to end the agreement with Acme?"
answer = agent.chat(question)
# The agent called the tool with the name "Termination notice period", and got
# the value "Either party must give 30 days' written notice to terminate the
# agreement for any reason.", with its source: the sentence of the document
# on termination.
# answer: "You need to give Acme **30 days' written notice** to end the
#     agreement.", followed by that sentence as a quote
