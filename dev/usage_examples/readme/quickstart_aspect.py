# Quick Start Example - Extracting payment terms from a document

import os

from contextgem import Aspect, Document, DocumentLLM

# Sample document text (shortened for brevity)
doc = Document(
    raw_text=(
        "SERVICE AGREEMENT\n"
        "SERVICES. Provider agrees to provide the following services to Client: "
        "Cloud-based data analytics platform access and maintenance...\n"
        "PAYMENT. Client agrees to pay $5,000 per month for the services. "
        "Payment is due on the 1st of each month. Late payments will incur a 2% fee per month...\n"
        "CONFIDENTIALITY. Both parties agree to keep all proprietary information confidential "
        "for a period of 5 years following termination of this Agreement..."
    ),
)

# Define the aspects to extract
doc.aspects = [
    Aspect(
        name="Payment Terms",
        description="Payment terms and conditions in the contract",
        # see the docs for more configuration options, e.g. sub-aspects, concepts, etc.
    ),
    # Add more aspects as needed
]
# Or use `doc.add_aspects([...])`

# Define an LLM for extracting information from the document
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or another provider/LLM
    api_key=os.environ.get(
        "CONTEXTGEM_OPENAI_API_KEY"
    ),  # your API key for the LLM provider
    # see the docs for more configuration options
)

# Extract information from the document
doc = llm.extract_all(doc)  # or use async version `await llm.extract_all_async(doc)`

# Access extracted information in the document object
for item in doc.aspects[0].extracted_items:
    print(f"• {item.value}")
# or `doc.get_aspect_by_name("Payment Terms").extracted_items`

# Output (paragraph from the document):
# • PAYMENT. Client agrees to pay $5,000 per month for the services. Payment is due on the 1st of each month. Late payments will incur a 2% fee per month...
