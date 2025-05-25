# ContextGem: Aspect Extraction

import os

from contextgem import Aspect, Document, DocumentLLM

# Create a document instance
doc = Document(
    raw_text=(
        "Software License Agreement\n"
        "This software license agreement (Agreement) is entered into between Tech Corp (Licensor) and Client Corp (Licensee).\n"
        "...\n"
        "2. Term and Termination\n"
        "This Agreement shall commence on the Effective Date and shall continue for a period of three (3) years, "
        "unless earlier terminated in accordance with the provisions hereof. Either party may terminate this Agreement "
        "upon thirty (30) days written notice to the other party.\n"
        "\n"
        "3. Payment Terms\n"
        "Licensee agrees to pay Licensor an annual license fee of $10,000, payable within thirty (30) days of the "
        "invoice date. Late payments shall incur a penalty of 1.5% per month.\n"
        "...\n"
    ),
)

# Define an aspect to extract the termination clause
termination_aspect = Aspect(
    name="Termination Clauses",
    description="Sections describing how and when the agreement can be terminated, including notice periods and conditions",
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

# Extract the aspect from the document
termination_aspect = llm.extract_aspects_from_document(doc)[0]

# Access the extracted information
print("Extracted Termination Clauses:")
for item in termination_aspect.extracted_items:
    print(f"- {item.value}")
