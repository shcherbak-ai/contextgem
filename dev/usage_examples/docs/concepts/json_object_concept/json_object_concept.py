# ContextGem: JsonObjectConcept Extraction

import os
from pprint import pprint
from typing import Literal

from contextgem import Document, DocumentLLM, JsonObjectConcept

# Define product information text
product_text = """
Product: Smart Fitness Watch X7
Price: $199.99
Features: Heart rate monitoring, GPS tracking, Sleep analysis
Battery Life: 5 days
Water Resistance: IP68
Available Colors: Black, Silver, Blue
Customer Rating: 4.5/5
"""

# Create a Document object from text
doc = Document(raw_text=product_text)

# Define a JsonObjectConcept with a structure for product information
product_concept = JsonObjectConcept(
    name="Product Information",
    description="Extract detailed product information including name, price, features, and specifications",
    structure={
        "name": str,
        "price": float,
        "features": list[str],
        "specifications": {
            "battery_life": str,
            "water_resistance": Literal["IP67", "IP68", "IPX7", "Not water resistant"],
        },
        "available_colors": list[str],
        "customer_rating": float,
    },
)

# Attach the concept to the document
doc.add_concepts([product_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document
product_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted structured data
extracted_product = product_concept.extracted_items[0].value
pprint(extracted_product)
