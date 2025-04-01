# Quick Start Example - Extracting an aspect with sub-aspects

import os

from contextgem import Aspect, Document, DocumentLLM

# Sample document (content shortened for brevity)
contract_text = """
EMPLOYMENT AGREEMENT
...
8. TERMINATION
8.1 Termination by the Company. The Company may terminate the Employee's employment for Cause at any time upon written notice. 
"Cause" shall mean: (i) Employee's material breach of this Agreement; (ii) Employee's conviction of a felony; or 
(iii) Employee's willful misconduct that causes material harm to the Company.
8.2 Termination by the Employee. The Employee may terminate employment for Good Reason upon 30 days' written notice to the Company. 
"Good Reason" shall mean a material reduction in Employee's base salary or a material diminution in Employee's duties.
8.3 Severance. If the Employee's employment is terminated by the Company without Cause or by the Employee for Good Reason, 
the Employee shall be entitled to receive severance pay equal to six (6) months of the Employee's base salary.
...
"""

doc = Document(raw_text=contract_text)

# Define termination aspect with practical sub-aspects
termination_aspect = Aspect(
    name="Termination",
    description="Provisions related to the termination of employment",
    aspects=[  # assign sub-aspects (optional)
        Aspect(
            name="Company Termination Rights",
            description="Conditions under which the company can terminate employment",
        ),
        Aspect(
            name="Employee Termination Rights",
            description="Conditions under which the employee can terminate employment",
        ),
        Aspect(
            name="Severance Terms",
            description="Compensation or benefits provided upon termination",
        ),
    ],
)

# Add the aspect to the document. Sub-aspects are added with the parent aspect.
doc.add_aspects([termination_aspect])
# (add more aspects to the document, if needed)

# Create an LLM for extraction
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or any other LLM from e.g. Anthropic, etc.
    api_key=os.environ.get(
        "CONTEXTGEM_OPENAI_API_KEY"
    ),  # your API key of the LLM provider
)

# Extract all information from the document
doc = llm.extract_all(doc)

# Get results with references in the document object
print("\nTermination aspect:\n")
termination_aspect = doc.get_aspect_by_name("Termination")
for sub_aspect in termination_aspect.aspects:
    print(sub_aspect.name)
    for item in sub_aspect.extracted_items:
        print(item.value)
    print("\n")
