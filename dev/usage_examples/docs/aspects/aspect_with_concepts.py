# ContextGem: Aspect Extraction with Concepts

import os

from contextgem import Aspect, Document, DocumentLLM, NumericalConcept, StringConcept

# Create a document instance
doc = Document(
    raw_text=(
        "Service Agreement\n"
        "This Service Agreement is between DataFlow Solutions (Provider) and Enterprise Corp (Client).\n"
        "\n"
        "3. Payment Terms\n"
        "3.1 Service Fees\n"
        "The Client shall pay the Provider a monthly service fee of $5,000 for basic services. "
        "Additional premium features are available for an extra $1,200 per month. "
        "Setup fee is a one-time payment of $2,500.\n"
        "\n"
        "3.2 Payment Schedule\n"
        "All payments are due within 15 business days of invoice receipt. "
        "Invoices will be sent on the first day of each month for the upcoming service period. "
        "Late payments will incur a penalty of 2% per month on the outstanding balance.\n"
        "\n"
        "3.3 Payment Methods\n"
        "Payments may be made by bank transfer, corporate check, or ACH. "
        "Credit card payments are accepted for amounts under $1,000 with a 3% processing fee. "
        "Wire transfer fees are the responsibility of the Client.\n"
        "\n"
        "3.4 Refund Policy\n"
        "Services are non-refundable once delivered. However, if services are terminated "
        "with 30 days notice, any prepaid fees for future periods will be refunded on a pro-rata basis.\n"
    ),
)

# Define an aspect with associated concepts
payment_aspect = Aspect(
    name="Payment Terms",
    description="All clauses and provisions related to payment, including fees, schedules, methods, and policies",
    concepts=[
        NumericalConcept(
            name="Monthly Service Fee",
            description="The regular monthly fee for basic services",
            numeric_type="float",
        ),
        NumericalConcept(
            name="Premium Features Fee",
            description="Additional monthly fee for premium features",
            numeric_type="float",
        ),
        NumericalConcept(
            name="Setup Fee",
            description="One-time initial setup or onboarding fee",
            numeric_type="float",
        ),
        NumericalConcept(
            name="Payment Due Days",
            description="Number of days the client has to make payment after receiving invoice",
            numeric_type="int",
        ),
        NumericalConcept(
            name="Late Payment Penalty Rate",
            description="Percentage penalty charged per month for late payments",
            numeric_type="float",
        ),
        StringConcept(
            name="Accepted Payment Methods",
            description="List of payment methods that are accepted by the provider",
        ),
        StringConcept(
            name="Refund Policy",
            description="Conditions and procedures for refunds or credits",
        ),
    ],
)

# Add the aspect to the document
doc.add_aspects([payment_aspect])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract aspects and their concepts from the document
doc = llm.extract_all(doc)

# Access the extracted payment terms aspect and concepts
payment_terms_aspect = doc.get_aspect_by_name("Payment Terms")
print("Extracted Payment Terms Section:")
for item in payment_terms_aspect.extracted_items:
    print(f"- {item.value}")
print("\nExtracted Payment Details:")
for concept in payment_terms_aspect.concepts:
    print(f"\n{concept.name}:")
    for item in concept.extracted_items:
        print(f"- {item.value}")

# Access specific extracted values
monthly_fee = payment_terms_aspect.get_concept_by_name("Monthly Service Fee")
print(f"\nMonthly Service Fee: ${monthly_fee.extracted_items[0].value}")
