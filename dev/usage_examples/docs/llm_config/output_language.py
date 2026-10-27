# Output in Japanese from a contract in German with an amendment in French

import os

from contextgem import Document, DocumentLLM, StringConcept


doc = Document(
    raw_text=(
        "Liefervertrag\n\n"
        "§ 1 Die Nordwerk GmbH liefert der Lumière SARL Pumpen des Typs HX-200.\n\n"
        "§ 2 Der Vertrag gilt für drei (3) Jahre ab Unterzeichnung.\n\n"
        "§ 3 Es gilt deutsches Recht. Gerichtsstand ist München.\n\n"
        "Avenant n° 1\n\n"
        "Par dérogation au § 2, le contrat est conclu pour une durée de cinq (5) ans "
        "à compter de sa signature."
    )
)

doc.add_concepts(
    [
        StringConcept(
            name="Term of the agreement",
            description="How long the agreement lasts, as amended",
            add_justifications=True,
        ),
        StringConcept(
            name="Governing law",
            description="The law that governs the agreement, and the court",
        ),
    ]
)

llm = DocumentLLM(
    model="anthropic/claude-haiku-5-5",
    api_key=os.environ.get("CONTEXTGEM_ANTHROPIC_API_KEY"),
    # A BCP 47 language tag, e.g. "ja", "zh-Hant" or "pt-BR": values and
    # justifications in that language, whatever the languages of the document
    output_language="ja",
)

doc = llm.extract_all(doc)  # or use async version `await llm.extract_all_async(doc)`

for concept in doc.concepts:
    for item in concept.extracted_items:
        print(f"{concept.name}: {item.value}")
        if item.justification:
            print(f"  {item.justification}")
