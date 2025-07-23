# ContextGem: RatingConcept Extraction

import os

from contextgem import Document, DocumentLLM, RatingConcept


# Create a Document object from text describing a product without an explicit rating
smartphone_description = (
    "This smartphone features a 5000mAh battery that lasts all day with heavy use. "
    "The display is 6.7 inch AMOLED with 120Hz refresh rate. "
    "Camera system includes a 50MP main sensor, 12MP ultrawide, and 8MP telephoto lens. "
    "The phone runs on the latest processor with 8GB RAM and 256GB storage. "
    "It has IP68 water resistance and Gorilla Glass Victus protection."
)

doc = Document(raw_text=smartphone_description)

# Define a RatingConcept that requires analysis to determine a rating
product_quality = RatingConcept(
    name="Product Quality Rating",
    description=(
        "Evaluate the overall quality of the smartphone based on its specifications, "
        "features, and adherence to industry best practices"
    ),
    rating_scale=(1, 10),
    add_justifications=True,  # include justification for the rating
    justification_depth="balanced",
    justification_max_sents=5,
)

# Attach the concept to the document
doc.add_concepts([product_quality])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept from the document - the LLM will analyze and assign a rating
product_quality = llm.extract_concepts_from_document(doc)[0]

# Print the calculated rating
print(f"Quality Rating: {product_quality.extracted_items[0].value}")
# Print the justification
print(f"Justification: {product_quality.extracted_items[0].justification}")
