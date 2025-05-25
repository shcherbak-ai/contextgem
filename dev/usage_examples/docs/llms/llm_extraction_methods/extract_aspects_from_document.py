# ContextGem: Extracting Aspects from Documents

import os

from contextgem import Aspect, Document, DocumentLLM

# Sample text content
text_content = """
TechCorp is a leading software development company founded in 2015 with headquarters in San Francisco.
The company specializes in cloud-based solutions and has grown to 500 employees across 12 countries.
Their flagship product, CloudManager Pro, serves over 10,000 enterprise clients worldwide.
TechCorp reported $50 million in revenue for 2023, representing a 25% growth from the previous year.
The company is known for its innovative AI-powered analytics platform and excellent customer support.
They recently expanded into the European market and plan to launch three new products in 2024.
"""

# Create a Document object from text
doc = Document(raw_text=text_content)

# Define aspects to extract from the document
doc.aspects = [
    Aspect(
        name="Company Overview",
        description="Basic information about the company, founding, location, and size",
    ),
    Aspect(
        name="Financial Performance",
        description="Revenue, growth metrics, and financial indicators",
    ),
    Aspect(
        name="Products and Services",
        description="Information about the company's products, services, and offerings",
    ),
]

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract aspects from the document
extracted_aspects = llm.extract_aspects_from_document(doc)

# Access extracted aspect information
for aspect in extracted_aspects:
    print(f"Aspect: {aspect.name}")
    print(f"Extracted items: {[item.value for item in aspect.extracted_items]}")
    print("---")
