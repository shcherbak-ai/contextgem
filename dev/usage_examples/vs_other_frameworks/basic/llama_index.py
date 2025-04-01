# LlamaIndex implementation for extracting anomalies from a document, with source references and justifications

import os
from textwrap import dedent
from typing import Optional

from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field


# Pydantic models must be manually defined
class Anomaly(BaseModel):
    """An anomaly found in the document."""

    text: str = Field(description="The anomalous text found in the document")
    justification: str = Field(
        description="Brief justification for why this is an anomaly"
    )
    reference: str = Field(
        description="The sentence containing the anomaly"
    )  # LLM reciting a reference is error-prone and unreliable


class AnomaliesList(BaseModel):
    """List of anomalies found in the document."""

    anomalies: list[Anomaly] = Field(
        description="List of anomalies found in the document"
    )


def extract_anomalies_with_llama_index(
    document_text: str, api_key: Optional[str] = None
) -> list[Anomaly]:
    """
    Extract anomalies from a document using LlamaIndex.

    Args:
        document_text: The text content of the document
        api_key: OpenAI API key (defaults to environment variable)

    Returns:
        List of extracted anomalies with justifications and references
    """
    openai_api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY")
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        """
    You are an expert document analyzer. Your task is to identify any anomalies in the document.
    Anomalies are statements, phrases, or content that seem out of place, irrelevant, or inconsistent
    with the rest of the document's context and purpose.
    
    Document:
    {document_text}
    
    Identify all anomalies in the document. For each anomaly, provide:
    1. The anomalous text
    2. A brief justification explaining why it's an anomaly
    3. The complete sentence containing the anomaly for reference
    """
    )

    # Use PydanticOutputParser to directly parse the LLM output into our structured format
    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=AnomaliesList),
        prompt_template_str=prompt_template,
        llm=llm,
        verbose=True,
    )

    # Execute the program
    try:
        result = program(document_text=document_text)
        return result.anomalies
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return []


# Example usage
# Sample document text (shortened for brevity)
document_text = (
    "Consultancy Agreement\n"
    "This agreement between Company A (Supplier) and Company B (Customer)...\n"
    "The term of the agreement is 1 year from the Effective Date...\n"
    "The Supplier shall provide consultancy services as described in Annex 2...\n"
    "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
    "The purple elephant danced gracefully on the moon while eating ice cream.\n"  # out-of-context / anomaly
    "This agreement is governed by the laws of Norway...\n"
)

# Extract anomalies
anomalies = extract_anomalies_with_llama_index(document_text)

# Print results
for anomaly in anomalies:
    print(f"Anomaly: {anomaly}")
