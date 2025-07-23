# ContextGem: JsonObjectConcept Extraction with References and Justifications

import os
from pprint import pprint
from typing import Literal

from contextgem import Document, DocumentLLM, JsonObjectConcept


# Sample document text containing a customer complaint
customer_complaint = """
CUSTOMER COMPLAINT #CR-2023-0472
Date: November 15, 2023
Customer: Sarah Johnson

Description:
I purchased the Ultra Premium Blender (Model XJ-5000) from your online store on October 3, 2023. The product was delivered on October 10, 2023. After using it only 5 times, the motor started making loud grinding noises and then completely stopped working on November 12.

I've tried troubleshooting using the manual, including checking for obstructions and resetting the device, but nothing has resolved the issue. I expected much better quality given the premium price point ($249.99) and the 5-year warranty advertised.

I've been a loyal customer for over 7 years and have purchased several kitchen appliances from your company. This is the first time I've experienced such a significant quality issue. I would like a replacement unit or a full refund.

Previous interactions:
- Spoke with customer service representative Alex on Nov 13 (Ref #CS-98721)
- Was told to submit this formal complaint after troubleshooting was unsuccessful
- No resolution offered during initial call

Contact: sarah.johnson@example.com | (555) 123-4567
"""

# Create a Document from the text
doc = Document(raw_text=customer_complaint)

# Create a JsonObjectConcept with justifications and references enabled
complaint_analysis_concept = JsonObjectConcept(
    name="Complaint analysis",
    description="Detailed analysis of a customer complaint",
    structure={
        "issue_type": Literal[
            "product defect",
            "delivery problem",
            "billing error",
            "service issue",
            "other",
        ],
        "warranty_applicable": bool,
        "severity": Literal["low", "medium", "high", "critical"],
        "customer_loyalty_status": Literal["new", "regular", "loyal", "premium"],
        "recommended_resolution": Literal[
            "replacement", "refund", "repair", "partial refund", "other"
        ],
        "priority_level": Literal["low", "standard", "high", "urgent"],
        "expected_business_impact": Literal["minimal", "moderate", "significant"],
    },
    add_justifications=True,
    justification_depth="comprehensive",  # provide detailed justifications
    justification_max_sents=10,  # provide up to 10 sentences for each justification
    add_references=True,
    reference_depth="sentences",  # provide references to the sentences in the document
)

# Attach the concept to the document
doc.add_concepts([complaint_analysis_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept
complaint_analysis_concept = llm.extract_concepts_from_document(doc)[0]

# Get the extracted complaint analysis
complaint_analysis_item = complaint_analysis_concept.extracted_items[0]

# Print the structured analysis
print("Complaint Analysis\n")
pprint(complaint_analysis_item.value)

print("\nJustification:")
print(complaint_analysis_item.justification)

# Print key source references
print("\nReferences:")
for sent in complaint_analysis_item.reference_sentences:
    print(f"- {sent.raw_text}")
