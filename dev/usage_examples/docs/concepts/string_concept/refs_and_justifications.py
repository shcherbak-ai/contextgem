# ContextGem: StringConcept Extraction with References and Justifications

import os

from contextgem import Document, DocumentLLM, StringConcept

# Sample document text containing financial information
financial_text = """
2024 Financial Performance Summary

Revenue increased to $120 million in fiscal year 2024, representing 15% growth compared to the previous year. This growth was primarily driven by the expansion of our enterprise client base and the successful launch of our premium service tier.

The Board has recommended a dividend of $1.25 per share, which will be payable to shareholders of record as of March 15, 2025.
"""

# Create a Document from the text
doc = Document(raw_text=financial_text)

# Create a StringConcept with justifications and references enabled
key_figures_concept = StringConcept(
    name="Financial key figures",
    description="Important financial metrics and figures mentioned in the report",
    add_justifications=True,  # enable justifications to understand extraction reasoning
    justification_depth="balanced",
    justification_max_sents=3,  # allow up to 3 sentences for each justification
    add_references=True,  # include references to source text
    reference_depth="sentences",  # reference specific sentences rather than paragraphs
)

# Attach the concept to the document
doc.add_concepts([key_figures_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4o-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept
key_figures_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted items with justifications and references
print("Extracted financial key figures:")
for item in key_figures_concept.extracted_items:
    print(f"\nFigure: {item.value}")
    print(f"Justification: {item.justification}")
    print("Source references:")
    for sent in item.reference_sentences:
        print(f"- {sent.raw_text}")
