# ContextGem: Aspect Analysis with LabelConcept

import os

from contextgem import Aspect, Document, DocumentLLM, LabelConcept

# Create a Document object from contract text
contract_text = """
SOFTWARE DEVELOPMENT AGREEMENT
...

SECTION 5. PAYMENT TERMS
Client shall pay Developer a total fee of $150,000 for the complete software development project, payable in three installments: $50,000 upon signing, $50,000 at milestone completion, and $50,000 upon final delivery.
...

SECTION 8. MAINTENANCE AND SUPPORT
Following project completion, Developer shall provide 12 months of maintenance and support services at a rate of $5,000 per month, totaling $60,000 annually.
...

SECTION 12. PENALTY CLAUSES
In the event of project delay beyond the agreed timeline, Developer shall pay liquidated damages of $2,000 per day of delay, with a maximum penalty cap of $50,000.
...

SECTION 15. INTELLECTUAL PROPERTY LICENSING
Client agrees to pay ongoing licensing fees of $10,000 annually for the use of Developer's proprietary frameworks and libraries integrated into the software solution.
...

SECTION 18. TERMINATION COSTS
Should Client terminate this agreement without cause, Client shall pay Developer 75% of all remaining unpaid fees, estimated at approximately $100,000 based on current project status.
...
"""

doc = Document(raw_text=contract_text)

# Define a LabelConcept to classify the financial risk level of the obligations
risk_classification_concept = LabelConcept(
    name="Client Financial Risk Level",
    description=(
        "Classify the financial risk level for the Client's financial obligations based on:\n"
        "- Amount size and impact on Client's cash flow\n"
        "- Payment timing and predictability for the Client\n"
        "- Penalty or liability exposure for the Client\n"
        "- Ongoing vs. one-time obligations for the Client"
    ),
    labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    classification_type="multi_class",
    add_justifications=True,
    justification_depth="comprehensive",  # provide a comprehensive justification
    justification_max_sents=10,  # set an adequate justification length
    singular_occurrence=True,  # global risk level for the client's financial obligations
)

# Define Aspect containing the concept
financial_obligations_aspect = Aspect(
    name="Client Financial Obligations",
    description="Financial obligations that the Client must fulfill under the contract",
    concepts=[risk_classification_concept],
)

# Attach the aspect to the document
doc.add_aspects([financial_obligations_aspect])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract all data from the document
doc = llm.extract_all(doc)

# Get the extracted aspect and concept
financial_obligations_aspect = doc.get_aspect_by_name(
    "Client Financial Obligations"
)  # or `doc.aspects[0]`
risk_classification_concept = financial_obligations_aspect.get_concept_by_name(
    "Client Financial Risk Level"
)  # or `financial_obligations_aspect.concepts[0]`

# Display the extracted information

print("Extracted Client Financial Obligations:")
for extracted_item in financial_obligations_aspect.extracted_items:
    print(f"- {extracted_item.value}")

if risk_classification_concept.extracted_items:
    assert (
        len(risk_classification_concept.extracted_items) == 1
    )  # as we have set `singular_occurrence=True` on the concept
    risk_item = risk_classification_concept.extracted_items[0]
    print(f"\nClient Financial Risk Level: {risk_item.value[0]}")
    print(f"Justification: {risk_item.justification}")
else:
    print("\nRisk level could not be determined")
