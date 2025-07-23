# ContextGem: Complex Hierarchical Aspect Extraction with Sub-Aspects and Concepts

import os

from contextgem import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentLLM,
    NumericalConcept,
    StringConcept,
)


# Create a document instance
doc = Document(
    raw_text=(
        "Software Development and Licensing Agreement\n"
        "\n"
        "1. Intellectual Property Rights\n"
        "1.1 Ownership of Developed Software\n"
        "All software developed under this Agreement shall remain the exclusive property of the Developer. "
        "The Client receives a non-exclusive license to use the software as specified in Section 2.\n"
        "\n"
        "1.2 Client Data and Content\n"
        "The Client retains all rights to data and content provided to the Developer. "
        "The Developer may not use Client data for any purpose other than fulfilling this Agreement.\n"
        "\n"
        "1.3 Third-Party Components\n"
        "The software may include third-party open-source components. The Client agrees to comply "
        "with all applicable open-source licenses.\n"
        "\n"
        "2. License Terms\n"
        "2.1 Grant of License\n"
        "Developer grants Client a perpetual, non-transferable license to use the software "
        "for internal business purposes only, limited to 100 concurrent users.\n"
        "\n"
        "2.2 License Restrictions\n"
        "Client may not redistribute, sublicense, or create derivative works. "
        "Reverse engineering is prohibited except as required by law.\n"
        "\n"
        "3. Payment and Financial Terms\n"
        "3.1 Development Fees\n"
        "Total development fee is $150,000, payable in three installments: "
        "$50,000 upon signing, $50,000 at 50% completion, and $50,000 upon delivery.\n"
        "\n"
        "3.2 Ongoing License Fees\n"
        "Annual license fee of $12,000 is due each year starting from the first anniversary. "
        "Fees may increase by up to 5% annually with 60 days notice.\n"
        "\n"
        "3.3 Payment Terms\n"
        "All payments due within 30 days of invoice. Late payments incur 1.5% monthly penalty.\n"
        "\n"
        "4. Liability and Risk Allocation\n"
        "4.1 Limitation of Liability\n"
        "Developer's total liability shall not exceed the total amount paid under this Agreement. "
        "Neither party shall be liable for indirect, consequential, or punitive damages.\n"
        "\n"
        "4.2 Indemnification\n"
        "Client agrees to indemnify Developer against third-party claims arising from Client's use "
        "of the software, except for claims related to Developer's IP infringement.\n"
        "\n"
        "4.3 Insurance Requirements\n"
        "Developer shall maintain professional liability insurance of at least $1,000,000. "
        "Client shall maintain general liability insurance of at least $2,000,000.\n"
    ),
)

# Define a complex hierarchical structure
contract_aspects = [
    Aspect(
        name="Intellectual Property Provisions",
        description="All provisions related to intellectual property rights, ownership, and usage",
        aspects=[
            Aspect(
                name="Software Ownership",
                description="Clauses defining who owns the developed software and related IP rights",
                concepts=[
                    StringConcept(
                        name="Software Owner",
                        description="The party that owns the developed software",
                    ),
                    BooleanConcept(
                        name="Exclusive Ownership",
                        description="Whether the ownership is exclusive to one party",
                    ),
                ],
            ),
            Aspect(
                name="Client Data Rights",
                description="Provisions about client data ownership and developer's permitted use",
                concepts=[
                    StringConcept(
                        name="Data Usage Restrictions",
                        description="Limitations on how developer can use client data",
                    ),
                ],
            ),
            Aspect(
                name="Third-Party Components",
                description="Terms regarding use of third-party or open-source components",
                concepts=[
                    BooleanConcept(
                        name="Open Source Included",
                        description="Whether the software includes open-source components",
                    ),
                ],
            ),
        ],
    ),
    Aspect(
        name="License Grant and Restrictions",
        description="Terms defining the software license granted to the client and any restrictions",
        aspects=[
            Aspect(
                name="License Scope",
                description="The extent and limitations of the license granted",
                concepts=[
                    StringConcept(
                        name="License Type",
                        description="The type of license granted (exclusive, non-exclusive, etc.)",
                    ),
                    NumericalConcept(
                        name="User Limit",
                        description="Maximum number of concurrent users allowed",
                        numeric_type="int",
                    ),
                    BooleanConcept(
                        name="Perpetual License",
                        description="Whether the license is perpetual or time-limited",
                    ),
                ],
            ),
            Aspect(
                name="Usage Restrictions",
                description="Prohibited uses and activities under the license",
                concepts=[
                    BooleanConcept(
                        name="Redistribution Allowed",
                        description="Whether client can redistribute the software",
                    ),
                    BooleanConcept(
                        name="Derivative Works Allowed",
                        description="Whether client can create derivative works",
                    ),
                ],
            ),
        ],
    ),
    Aspect(
        name="Financial Terms",
        description="All payment-related provisions including fees, schedules, and penalties",
        concepts=[
            NumericalConcept(
                name="Total Development Fee",
                description="The total amount for software development",
                numeric_type="float",
            ),
            NumericalConcept(
                name="Annual License Fee",
                description="Yearly fee for using the software",
                numeric_type="float",
            ),
            NumericalConcept(
                name="Payment Due Days",
                description="Number of days to make payment after invoice",
                numeric_type="int",
            ),
        ],
    ),
    Aspect(
        name="Risk and Liability Management",
        description="Provisions for managing risks, liability limitations, and insurance requirements",
        aspects=[
            Aspect(
                name="Liability Limitations",
                description="Caps and exclusions on each party's liability",
                concepts=[
                    StringConcept(
                        name="Liability Cap",
                        description="Maximum amount of liability for each party",
                    ),
                    StringConcept(
                        name="Excluded Damages",
                        description="Types of damages that are excluded from liability",
                    ),
                ],
            ),
            Aspect(
                name="Insurance Requirements",
                description="Required insurance coverage for each party",
                concepts=[
                    NumericalConcept(
                        name="Developer Insurance Amount",
                        description="Minimum professional liability insurance for developer",
                        numeric_type="float",
                    ),
                    NumericalConcept(
                        name="Client Insurance Amount",
                        description="Minimum general liability insurance for client",
                        numeric_type="float",
                    ),
                ],
            ),
        ],
    ),
]

# Add all aspects to the document
doc.add_aspects(contract_aspects)

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract aspects and concepts
doc = llm.extract_all(doc)

# Access the hierarchical extraction results
print("=== CONTRACT ANALYSIS RESULTS ===\n")

for main_aspect in doc.aspects:
    print(f"{main_aspect.name.upper()}")
    for item in main_aspect.extracted_items:
        print(f"- {item.value}")

    # Access main aspect concepts
    if main_aspect.concepts:
        print("  Main Aspect Concepts:")
        for concept in main_aspect.concepts:
            print(f"    • {concept.name}:")
            for item in concept.extracted_items:
                print(f"      - {item.value}")

    # Access sub-aspects
    if main_aspect.aspects:
        print("  Sub-Aspects:")
        for sub_aspect in main_aspect.aspects:
            print(f"    {sub_aspect.name}")
            for item in sub_aspect.extracted_items:
                print(f"    - {item.value}")

            # Access sub-aspect concepts
            if sub_aspect.concepts:
                print("    Sub-Aspect Concepts:")
                for concept in sub_aspect.concepts:
                    print(f"      • {concept.name}:")
                    for item in concept.extracted_items:
                        print(f"        - {item.value}")

    print()
