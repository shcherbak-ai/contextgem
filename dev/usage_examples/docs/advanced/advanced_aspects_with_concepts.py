# Advanced Usage Example - extracting a single aspect with inner concepts from a legal document

import os

from contextgem import Aspect, Document, DocumentLLM, StringConcept, StringExample


# Create a document instance with e.g. a legal contract text
# The text is shortened for brevity
doc = Document(
    raw_text=(
        "EMPLOYMENT AGREEMENT\n\n"
        'This Employment Agreement (the "Agreement") is made and entered into as of January 15, 2023 (the "Effective Date"), '
        'by and between ABC Corporation, a Delaware corporation (the "Company"), and Jane Smith, an individual (the "Employee").\n\n'
        "1. EMPLOYMENT TERM\n"
        "The Company hereby employs the Employee, and the Employee hereby accepts employment with the Company, upon the terms and "
        "conditions set forth in this Agreement. The term of this Agreement shall commence on the Effective Date and shall continue "
        'for a period of two (2) years, unless earlier terminated in accordance with Section 8 (the "Term").\n\n'
        "2. POSITION AND DUTIES\n"
        "During the Term, the Employee shall serve as Chief Technology Officer of the Company, with such duties and responsibilities "
        "as are commensurate with such position.\n\n"
        "8. TERMINATION\n"
        "8.1 Termination by the Company. The Company may terminate the Employee's employment for Cause at any time upon written notice. "
        "\"Cause\" shall mean: (i) Employee's material breach of this Agreement; (ii) Employee's conviction of a felony; or "
        "(iii) Employee's willful misconduct that causes material harm to the Company.\n"
        "8.2 Termination by the Employee. The Employee may terminate employment for Good Reason upon 30 days' written notice to the Company. "
        "\"Good Reason\" shall mean a material reduction in Employee's base salary or a material diminution in Employee's duties.\n"
        "8.3 Severance. If the Employee's employment is terminated by the Company without Cause or by the Employee for Good Reason, "
        "the Employee shall be entitled to receive severance pay equal to six (6) months of the Employee's base salary.\n\n"
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.\n\n"
        "ABC CORPORATION\n\n"
        "By: ______________________\n"
        "Name: John Johnson\n"
        "Title: CEO\n\n"
        "EMPLOYEE\n\n"
        "______________________\n"
        "Jane Smith"
    )
)

# Define an aspect focused on termination clauses
termination_aspect = Aspect(
    name="Termination Provisions",
    description="Analysis of contract termination conditions, notice requirements, and severance terms.",
    reference_depth="paragraphs",
)

# Define concepts for the termination aspect
termination_for_cause = StringConcept(
    name="Termination for Cause",
    description="Conditions under which the company can terminate the employee for cause.",
    examples=[  # optional, examples help the LLM to understand the concept better
        StringExample(content="Employee may be terminated for misconduct"),
        StringExample(content="Termination for breach of contract"),
    ],
    add_references=True,
    reference_depth="sentences",
)
notice_period = StringConcept(
    name="Notice Period",
    description="Required notification period before employment termination.",
    add_references=True,
    reference_depth="sentences",
)
severance_terms = StringConcept(
    name="Severance Package",
    description="Compensation and benefits provided upon termination.",
    add_references=True,
    reference_depth="sentences",
)

# Add concepts to the aspect
termination_aspect.add_concepts([termination_for_cause, notice_period, severance_terms])

# Add the aspect to the document
doc.add_aspects([termination_aspect])

# Create an LLM for extracting data from the document
llm = DocumentLLM(
    model="openai/gpt-4o",  # You can use models from other providers as well, e.g. "anthropic/claude-3-5-sonnet"
    api_key=os.environ.get(
        "CONTEXTGEM_OPENAI_API_KEY"
    ),  # your API key for OpenAI or another LLM provider
)

# Extract all information from the document
doc = llm.extract_all(doc)

# Access the extracted information in the document object
print("=== Termination Provisions Analysis ===")
print(f"Extracted {len(doc.aspects[0].extracted_items)} items from the aspect")

# Access extracted aspect concepts in the document object
for concept in doc.aspects[0].concepts:
    print(f"--- {concept.name} ---")
    for item in concept.extracted_items:
        print(f"• {item.value}")
        print(f"  Reference sentences: {len(item.reference_sentences)}")
