# ContextGem: LabelConcept with References and Justifications

import os

from contextgem import Document, DocumentLLM, LabelConcept


# Create a Document with content that might be challenging to classify
mixed_content_text = """
QUARTERLY BUSINESS REVIEW AND POLICY UPDATES
GlobalTech Solutions Inc. - February 2025

EMPLOYMENT AGREEMENT AND CONFIDENTIALITY PROVISIONS

This Employment Agreement ("Agreement") is entered into between GlobalTech Solutions Inc. ("Company") and Sarah Johnson ("Employee") as of February 1, 2025.

EMPLOYMENT TERMS
Employee shall serve as Senior Software Engineer with responsibilities including software development, code review, and technical leadership. The position is full-time with an annual salary of $125,000.

CONFIDENTIALITY OBLIGATIONS
Employee acknowledges that during employment, they may have access to confidential information including proprietary algorithms, customer data, and business strategies. Employee agrees to maintain strict confidentiality of such information both during and after employment.

NON-COMPETE PROVISIONS
For a period of 12 months following termination, Employee agrees not to engage in any business activities that directly compete with Company's core services within the same geographic market.

INTELLECTUAL PROPERTY
All work products, inventions, and discoveries made during employment shall be the exclusive property of the Company.

ADDITIONAL INFORMATION:

FINANCIAL PERFORMANCE SUMMARY
Q4 2024 revenue exceeded projections by 12%, reaching $3.2M. Cost optimization initiatives reduced operational expenses by 8%. The board approved a $500K investment in new data analytics infrastructure for 2025.

PRODUCT LAUNCH TIMELINE
The AI-powered customer analytics platform will launch Q2 2025. Marketing budget allocated: $200K for digital campaigns. Expected customer acquisition target: 150 new enterprise clients in the first quarter post-launch.
"""

doc = Document(raw_text=mixed_content_text)

# Define a LabelConcept with justifications and references enabled
document_classification_concept = LabelConcept(
    name="Document Classification with Evidence",
    description="Classify this document type and provide reasoning for the classification",
    labels=[
        "Employment Contract",
        "NDA",
        "Consulting Agreement",
        "Service Agreement",
        "Partnership Agreement",
        "Other",
    ],
    classification_type="multi_class",
    add_justifications=True,  # enable justifications to understand classification reasoning
    justification_depth="comprehensive",  # provide detailed reasoning
    justification_max_sents=5,  # allow up to 5 sentences for justification
    add_references=True,  # include references to source text
    reference_depth="paragraphs",  # reference specific paragraphs that informed classification
    singular_occurrence=True,  # expect only one classification result
)

# Attach the concept to the document
doc.add_concepts([document_classification_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
document_classification_concept = llm.extract_concepts_from_document(doc)[0]

# Display the classification results with evidence
if document_classification_concept.extracted_items:
    item = document_classification_concept.extracted_items[0]

    print("=== DOCUMENT CLASSIFICATION RESULTS ===")
    print(f"Classification: {item.value[0]}")
    print("\nJustification:")
    print(f"{item.justification}")

    print("\nEvidence from document:")
    for i, paragraph in enumerate(item.reference_paragraphs, 1):
        print(f"{i}. {paragraph.raw_text}")

else:
    print("No classification could be determined - none of the predefined labels apply")

# This example demonstrates how justifications help explain why the LLM
# chose a specific classification and how references show which parts
# of the document informed that decision
