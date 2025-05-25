# ContextGem: RatingConcept Extraction with References and Justifications

import os

from contextgem import Document, DocumentLLM, RatingConcept, RatingScale

# Sample document text about a software product with various aspects
software_review = """
Software Review: ProjectManager Pro 5.0

User Interface: The interface is clean and modern, with intuitive navigation. New users can quickly find what they need without extensive training. The dashboard provides a comprehensive overview of project status.

Performance: The application loads quickly even with large projects. Resource-intensive operations like generating reports occasionally cause minor lag on older systems. The mobile app performs exceptionally well, even on limited bandwidth.

Features: Project templates are well-designed and cover most common project types. Task dependencies are easily managed, and the Gantt chart visualization is excellent. However, the software lacks advanced risk management tools that competitors offer.

Support: The documentation is comprehensive and well-organized. Customer service response time averages 4 hours, which is acceptable but not industry-leading. The knowledge base needs more video tutorials.
"""

# Create a Document from the text
doc = Document(raw_text=software_review)

# Create a RatingConcept with justifications and references enabled
usability_rating_concept = RatingConcept(
    name="Software usability rating",
    description="Evaluate the overall usability of the software on a scale of 1-10 based on UI design, intuitiveness, and learning curve",
    rating_scale=RatingScale(start=1, end=10),
    add_justifications=True,  # enable justifications to explain the rating
    justification_depth="comprehensive",  # provide detailed reasoning
    justification_max_sents=5,  # allow up to 5 sentences for justification
    add_references=True,  # include references to source text
    reference_depth="sentences",  # reference specific sentences rather than paragraphs
)

# Attach the concept to the document
doc.add_concepts([usability_rating_concept])

# Configure DocumentLLM with your API parameters
llm = DocumentLLM(
    model="azure/gpt-4.1",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
)

# Extract the concept
usability_rating_concept = llm.extract_concepts_from_document(doc)[0]

# Print the extracted rating item with justification and references
extracted_item = usability_rating_concept.extracted_items[0]
print(f"Software Usability Rating: {extracted_item.value}/10")
print(f"\nJustification: {extracted_item.justification}")
print("\nSource references:")
for sent in extracted_item.reference_sentences:
    print(f"- {sent.raw_text}")
