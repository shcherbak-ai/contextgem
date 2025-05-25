# ContextGem: Aspect Extraction with Sub-Aspects

import os

from contextgem import Aspect, Document, DocumentLLM

# Create a document instance
doc = Document(
    raw_text=(
        "Employment Agreement\n"
        "This Employment Agreement is entered into between Global Tech Inc. (Company) and John Smith (Employee).\n"
        "\n"
        "Section 8: Termination\n"
        "8.1 Termination by Company\n"
        "The Company may terminate this agreement at any time with or without cause by providing thirty (30) days "
        "written notice to the Employee. In case of termination for cause, no notice period is required.\n"
        "\n"
        "8.2 Termination by Employee\n"
        "The Employee may terminate this agreement by providing fourteen (14) days written notice to the Company. "
        "The Employee must complete all pending assignments before the termination date.\n"
        "\n"
        "8.3 Severance Benefits\n"
        "Upon termination without cause, the Employee shall receive severance pay equal to two (2) weeks of base salary "
        "for each year of service, with a minimum of four (4) weeks and a maximum of twenty-six (26) weeks. "
        "Severance benefits are contingent upon signing a release agreement.\n"
        "\n"
        "8.4 Return of Company Property\n"
        "Upon termination, the Employee must immediately return all Company property, including laptops, access cards, "
        "confidential documents, and any other materials belonging to the Company.\n"
        "\n"
        "Section 9: Non-Competition\n"
        "The Employee agrees not to engage in any business that competes with the Company for a period of twelve (12) "
        "months following termination of employment within a 50-mile radius of the Company's headquarters.\n"
    ),
)

# Define the main termination aspect with sub-aspects
termination_aspect = Aspect(
    name="Termination Provisions",
    description="All provisions related to employment termination including conditions, procedures, and consequences",
    aspects=[
        Aspect(
            name="Company Termination Rights",
            description="Conditions and procedures for the company to terminate the employee, including notice periods and cause requirements",
        ),
        Aspect(
            name="Employee Termination Rights",
            description="Conditions and procedures for the employee to terminate employment, including notice requirements and obligations",
        ),
        Aspect(
            name="Severance Benefits",
            description="Compensation and benefits provided to the employee upon termination, including calculation methods and conditions",
        ),
        Aspect(
            name="Post-Termination Obligations",
            description="Employee obligations that continue after termination, including property return and non-competition requirements",
        ),
    ],
)

# Add the aspect to the document
doc.add_aspects([termination_aspect])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract aspects from the document
termination_aspect = llm.extract_aspects_from_document(doc)[0]

# Access the extracted information
print("All Termination Provisions:")
for item in termination_aspect.extracted_items:
    print(f"- {item.value}")
print("\nSub-Aspects:")
for sub_aspect in termination_aspect.aspects:
    print(f"\n{sub_aspect.name}:")
    for item in sub_aspect.extracted_items:
        print(f"- {item.value}")
