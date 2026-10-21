import os

from docling.document_converter import DocumentConverter

from contextgem import Aspect, Document, DocumentLLM, JsonObjectConcept


# Convert the file with Docling, e.g. PDF, DOCX, PPTX, XLSX or HTML
result = DocumentConverter().convert("service_agreement.pdf")
# The paragraphs keep their place in the structure of the file and their pages
doc = Document.from_docling(result.document)

doc.add_aspects(
    [
        Aspect(
            name="Termination",
            description="Clauses on the term of the agreement and its termination",
        )
    ]
)
doc.add_concepts(
    [
        JsonObjectConcept(
            name="Fees",
            description="The monthly fee of each service",
            structure={"service": str, "monthly_fee_eur": float, "billing": str},
            add_references=True,
        )
    ]
)

llm = DocumentLLM(
    model="openai/gpt-6-luna",  # or another LLM, cloud or local
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
)
doc = llm.extract_all(doc)

termination = doc.get_aspect_by_name("Termination")
# .extracted_items: 3 items, the heading of section 3 and clauses 3.1 and 3.2
# .reference_page_numbers: [2]

fees = doc.get_concept_by_name("Fees").extracted_items
# 3 items, one for each row of the table of fees. The first one:
# .value: {"service": "Hosting", "monthly_fee_eur": 1200.0,
#     "billing": "In advance"}
# .reference_paragraphs: the table row, with the header of each column:
#     "Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance"
# .reference_page_numbers: [1]
