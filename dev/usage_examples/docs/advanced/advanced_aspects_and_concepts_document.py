# Advanced Usage Example - Extracting aspects and concepts from a document, with references,
# using concurrency

import os

from aiolimiter import AsyncLimiter

from contextgem import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentLLM,
    JsonObjectConcept,
    StringConcept,
)

# Example privacy policy document (shortened for brevity)
doc = Document(
    raw_text=(
        "Privacy Policy\n\n"
        "Last Updated: March 15, 2024\n\n"
        "1. Data Collection\n"
        "We collect various types of information from our users, including:\n"
        "- Personal information (name, email address, phone number)\n"
        "- Device information (IP address, browser type, operating system)\n"
        "- Usage data (pages visited, time spent on site)\n"
        "- Location data (with your consent)\n\n"
        "2. Data Usage\n"
        "We use your information to:\n"
        "- Provide and improve our services\n"
        "- Send you marketing communications (if you opt-in)\n"
        "- Analyze website performance\n"
        "- Comply with legal obligations\n\n"
        "3. Data Sharing\n"
        "We may share your information with:\n"
        "- Service providers (for processing payments and analytics)\n"
        "- Law enforcement (when legally required)\n"
        "- Business partners (with your explicit consent)\n\n"
        "4. Data Retention\n"
        "We retain personal data for 24 months after your last interaction with our services. "
        "Analytics data is kept for 36 months.\n\n"
        "5. User Rights\n"
        "You have the right to:\n"
        "- Access your personal data\n"
        "- Request data deletion\n"
        "- Opt-out of marketing communications\n"
        "- Lodge a complaint with supervisory authorities\n\n"
        "6. Contact Information\n"
        "For privacy-related inquiries, contact our Data Protection Officer at privacy@example.com\n"
    ),
)

# Define all document-level concepts in a single declaration
document_concepts = [
    BooleanConcept(
        name="Is Privacy Policy",
        description="Verify if this document is a privacy policy",
        singular_occurrence=True,  # explicitly enforce singular extracted item (optional)
    ),
    DateConcept(
        name="Last Updated Date",
        description="The date when the privacy policy was last updated",
        singular_occurrence=True,  # explicitly enforce singular extracted item (optional)
    ),
    StringConcept(
        name="Contact Information",
        description="Contact details for privacy-related inquiries",
        add_references=True,
        reference_depth="sentences",
    ),
]

# Define all aspects with their concepts in a single declaration
aspects = [
    Aspect(
        name="Data Collection",
        description="Information about what types of data are collected from users",
        concepts=[
            JsonObjectConcept(
                name="Collected Data Types",
                description="List of different types of data collected from users",
                structure={
                    "personal_info": list[str],
                    "technical_info": list[str],
                    "usage_info": list[str],
                },  # simply use a dictionary with type hints (including generic aliases and union types)
                add_references=True,
                reference_depth="sentences",
            )
        ],
    ),
    Aspect(
        name="Data Retention",
        description="Information about how long different types of data are retained",
        concepts=[
            JsonObjectConcept(
                name="Retention Periods",
                description="The durations for which different types of data are retained",
                structure={
                    "personal_info": str | None,
                    "technical_info": str | None,
                    "usage_info": str | None,
                },  # use `str | None` type hints to allow for None values if not specified
                add_references=True,
                reference_depth="sentences",
                singular_occurrence=True,  # explicitly enforce singular extracted item (optional)
            )
        ],
    ),
    Aspect(
        name="Data Subject Rights",
        description="Information about the rights users have regarding their data",
        concepts=[
            StringConcept(
                name="Data Subject Rights",
                description="Rights available to users regarding their personal data",
                add_references=True,
                reference_depth="sentences",
            )
        ],
    ),
]

# Add aspects and concepts to the document
doc.add_aspects(aspects)
doc.add_concepts(document_concepts)

# Create an LLM for extraction
llm = DocumentLLM(
    model="openai/gpt-4o",  # or another LLM from e.g. Anthropic, Ollama, etc.
    api_key=os.environ.get(
        "CONTEXTGEM_OPENAI_API_KEY"
    ),  # your API key for the applicable LLM provider
    async_limiter=AsyncLimiter(
        3, 3
    ),  # customize async limiter for concurrency (optional)
)

# Extract all information from the document, using concurrency
doc = llm.extract_all(doc, use_concurrency=True)

# Access / print extracted information on the document object

print("Document Concepts:")
for concept in doc.concepts:
    print(f"{concept.name}:")
    for item in concept.extracted_items:
        print(f"• {item.value}")
    print()

print("Aspects and Concepts:")
for aspect in doc.aspects:
    print(f"[{aspect.name}]")
    for item in aspect.extracted_items:
        print(f"• {item.value}")
    print()
    for concept in aspect.concepts:
        print(f"{concept.name}:")
        for item in concept.extracted_items:
            print(f"• {item.value}")
    print()
