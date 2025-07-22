# ContextGem: Multi-Label Classification with LabelConcept

import os

from contextgem import Document, DocumentLLM, LabelConcept


# Create a Document object with business document text covering multiple topics
business_doc_text = """
QUARTERLY BUSINESS REVIEW - Q4 2024

FINANCIAL PERFORMANCE
Revenue for Q4 2024 reached $2.8 million, exceeding our target by 12%. The finance team has prepared detailed budget projections for 2025, with anticipated growth of 18% across all divisions.

TECHNOLOGY INITIATIVES
Our development team has successfully implemented the new cloud infrastructure, reducing operational costs by 25%. The IT department is now focusing on cybersecurity enhancements and data analytics platform upgrades.

HUMAN RESOURCES UPDATE
We welcomed 15 new employees this quarter, bringing our total headcount to 145. The HR team has launched a comprehensive employee wellness program and updated our remote work policies.

LEGAL AND COMPLIANCE
All regulatory compliance requirements have been met for Q4. The legal department has reviewed and updated our data privacy policies in accordance with recent legislation changes.

MARKETING STRATEGY
The marketing team launched three successful campaigns this quarter, resulting in a 40% increase in lead generation. Our digital marketing efforts have expanded to include LinkedIn advertising and content marketing.
"""

doc = Document(raw_text=business_doc_text)

# Define a LabelConcept for topic classification allowing multiple topics
content_topics_concept = LabelConcept(
    name="Document Topics",
    description="Identify all relevant business topics covered in this document",
    labels=[
        "Finance",
        "Technology",
        "HR",
        "Legal",
        "Marketing",
        "Operations",
        "Sales",
        "Strategy",
    ],
    classification_type="multi_label",  # multiple labels can be selected (non-exclusive labels)
)


# Attach the concept to the document
doc.add_concepts([content_topics_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
content_topics_concept = llm.extract_concepts_from_document(doc)[0]

# Check if any labels were extracted
if content_topics_concept.extracted_items:
    # Get all identified topics
    identified_topics = content_topics_concept.extracted_items[0].value
    print(f"Document covers the following topics: {', '.join(identified_topics)}")
    # Expected output might include: Finance, Technology, HR, Legal, Marketing
else:
    print("No applicable topic labels found for this document")
