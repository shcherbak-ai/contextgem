# ContextGem: Aspect Extraction with Justifications

import os

from contextgem import Aspect, Document, DocumentLLM


# Create a document instance
doc = Document(
    raw_text=(
        "NON-DISCLOSURE AGREEMENT\n"
        "\n"
        'This Non-Disclosure Agreement ("Agreement") is entered into between TechCorp Inc. '
        '("Disclosing Party") and Innovation Labs LLC ("Receiving Party") on January 15, 2024.\n'
        "...\n"
    ),
)

# Define a single aspect focused on NDA direction with justifications
nda_direction_aspect = Aspect(
    name="NDA Direction",
    description="Provisions informing the NDA direction (whether mutual or one-way) and information flow between parties",
    add_justifications=True,
    justification_depth="balanced",
    justification_max_sents=4,
)

# Add the aspect to the document
doc.aspects = [nda_direction_aspect]

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the aspect with justifications
nda_direction_aspect = llm.extract_aspects_from_document(doc)[0]
for i, item in enumerate(nda_direction_aspect.extracted_items, 1):
    print(f"- {i}. {item.value}")
    print(f"  Justification: {item.justification}")
    print()
