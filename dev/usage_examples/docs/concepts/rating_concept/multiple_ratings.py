# ContextGem: Multiple RatingConcept Extraction

import os

from contextgem import Document, DocumentLLM, RatingConcept

# Sample document text about a restaurant review with multiple quality aspects to rate
restaurant_review = """
Restaurant Review: Bella Cucina

Atmosphere: The restaurant has a warm, inviting ambiance with soft lighting and comfortable seating. The décor is elegant without being pretentious, and the noise level allows for easy conversation.

Food Quality: The ingredients were fresh and high-quality. The pasta was perfectly cooked al dente, and the sauces were flavorful and well-balanced. The seafood dish had slightly overcooked shrimp, but the fish was excellent.

Service: Our server was knowledgeable about the menu and wine list. Water glasses were kept filled, and plates were cleared promptly. However, there was a noticeable delay between appetizers and main courses.

Value: Portion sizes were generous for the price point. The wine list offers selections at various price points, though markup is slightly higher than average for comparable restaurants in the area.
"""

# Create a Document from the text
doc = Document(raw_text=restaurant_review)

# Define a consistent rating scale to be used across all rating categories
restaurant_rating_scale = (1, 5)

# Define multiple rating concepts for different quality aspects of the restaurant
atmosphere_rating = RatingConcept(
    name="Atmosphere Rating",
    description="Rate the restaurant's atmosphere and ambiance",
    rating_scale=restaurant_rating_scale,
)

food_rating = RatingConcept(
    name="Food Quality Rating",
    description="Rate the quality, preparation, and taste of the food",
    rating_scale=restaurant_rating_scale,
)

service_rating = RatingConcept(
    name="Service Rating",
    description="Rate the efficiency, knowledge, and attentiveness of the service",
    rating_scale=restaurant_rating_scale,
)

value_rating = RatingConcept(
    name="Value Rating",
    description="Rate the value for money considering portion sizes and pricing",
    rating_scale=restaurant_rating_scale,
)

# Attach all concepts to the document
doc.add_concepts([atmosphere_rating, food_rating, service_rating, value_rating])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract all concepts from the document
extracted_concepts = llm.extract_concepts_from_document(doc)

# Print all ratings
print("Restaurant Ratings (1-5 scale):")
for concept in extracted_concepts:
    if concept.extracted_items:
        print(f"{concept.name}: {concept.extracted_items[0].value}/5")

# Calculate and print overall average rating
average_rating = sum(
    concept.extracted_items[0].value for concept in extracted_concepts
) / len(extracted_concepts)
print(f"\nOverall Rating: {average_rating:.1f}/5")
