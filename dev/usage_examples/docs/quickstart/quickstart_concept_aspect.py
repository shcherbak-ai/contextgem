# Quick Start Example - Extracting a concept from an aspect

import os

from contextgem import Aspect, Document, DocumentLLM, StringConcept, StringExample


# Example document instance
# Document content is shortened for brevity
doc = Document(
    raw_text=(
        "Employment Agreement\n"
        "This agreement between TechCorp Inc. (Employer) and Jane Smith (Employee)...\n"
        "The employment shall commence on January 15, 2023 and continue until terminated...\n"
        "The Employee shall work as a Senior Software Engineer reporting to the CTO...\n"
        "The Employee shall receive an annual salary of $120,000 paid monthly...\n"
        "The Employee is entitled to 20 days of paid vacation per year...\n"
        "The Employee agrees to a notice period of 30 days for resignation...\n"
        "This agreement is governed by the laws of California...\n"
    ),
)

# Define an aspect with a specific concept, using natural language
doc_aspect = Aspect(
    name="Compensation",
    description="Clauses defining the compensation and benefits for the employee",
    reference_depth="sentences",
)

# Define a concept within the aspect
aspect_concept = StringConcept(
    name="Annual Salary",
    description="The annual base salary amount specified in the employment agreement",
    examples=[  # optional
        StringExample(
            content="$X per year",  # guidance regarding format
        )
    ],
    add_references=True,
    reference_depth="sentences",
)

# Add the concept to the aspect
doc_aspect.add_concepts([aspect_concept])
# (add more concepts to the aspect, if needed)

# Add the aspect to the document
doc.add_aspects([doc_aspect])
# (add more aspects to the document, if needed)

# Create an LLM for extraction
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or any other LLM from e.g. Anthropic, etc.
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),  # your API key
)

# Extract information from the document
doc = llm.extract_all(doc)
# or use async version llm.extract_all_async(doc)

# Access extracted information in the document object
print("Compensation aspect:")
print(
    doc.get_aspect_by_name("Compensation").extracted_items
)  # extracted aspect items with references to sentences
print("Annual Salary concept:")
print(
    doc.get_aspect_by_name("Compensation")
    .get_concept_by_name("Annual Salary")
    .extracted_items
)  # extracted concept items with references to sentences
