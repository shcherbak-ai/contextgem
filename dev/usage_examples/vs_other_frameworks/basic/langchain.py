# LangChain implementation for extracting anomalies from a document, with source references and justifications

import os
from textwrap import dedent

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
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


def extract_anomalies_with_langchain(
    document_text: str, api_key: str | None = None
) -> list[Anomaly]:
    """
    Extract anomalies from a document using LangChain.

    Args:
        document_text: The text content of the document
        api_key: OpenAI API key (defaults to environment variable)

    Returns:
        List of extracted anomalies with justifications and references
    """
    openai_api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)

    # Create a parser for structured output
    parser = PydanticOutputParser(pydantic_object=AnomaliesList)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    template = dedent(
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
    
    {format_instructions}
    """
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["document_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create a runnable chain
    chain = (
        {"document_text": lambda x: x}
        | RunnablePassthrough.assign()
        | prompt
        | llm
        | RunnableLambda(lambda x: parser.parse(x.content))
    )

    # Run the chain and extract anomalies
    parsed_output = chain.invoke(document_text)

    return parsed_output.anomalies


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
anomalies = extract_anomalies_with_langchain(document_text)

# Print results
for anomaly in anomalies:
    print(f"Anomaly: {anomaly}")
