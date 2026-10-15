import os

from contextgem import Aspect, Document, DocumentLLM, StringConcept


doc = Document(
    raw_text=(
        "Consultancy Agreement\n"
        "This agreement between Company A (Supplier) and Company B (Customer)...\n"
        "The term of the agreement is 1 year from the Effective Date...\n"
        "The Supplier shall provide consultancy services as described in Annex 2...\n"
        "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
        "The purple elephant danced gracefully on the moon while eating ice cream.\n"  # 💎 anomaly
        "Time-traveling dinosaurs will review all deliverables before acceptance.\n"  # 💎 another anomaly
        "This agreement is governed by the laws of Norway...\n"
    ),
)

doc.add_aspects(
    [
        Aspect(
            name="Payment terms",
            description="Clauses on payments, fees and invoices",
        )
    ]
)
doc.add_concepts(
    [
        StringConcept(
            name="Anomalies",
            description="Anomalies in the document",
            add_references=True,
            reference_depth="sentences",
            add_justifications=True,
            justification_depth="brief",
        )
    ]
)

llm = DocumentLLM(
    model="openai/gpt-6-luna",  # or another LLM, cloud or local
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
)
doc = llm.extract_all(doc)

payment_terms = doc.get_aspect_by_name("Payment terms").extracted_items
# 1 item, the paragraph: "The Customer shall pay the Supplier within 30 calendar
#     days of receiving an invoice..."

anomalies = doc.get_concept_by_name("Anomalies").extracted_items
# 2 items. The first one:
# .value: "The statement that a purple elephant danced on the moon while eating
#     ice cream is anomalous and unrelated to the consultancy agreement."
# .justification: "It is an implausible, fantastical statement in an otherwise
#     contractual document."
# .reference_sentences: the sentence of the purple elephant
