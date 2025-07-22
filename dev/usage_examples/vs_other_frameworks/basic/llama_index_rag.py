# LlamaIndex (RAG) implementation for extracting anomalies from a document, with source references and justifications

import os
from textwrap import dedent
from typing import Any

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field


# Pydantic models must be manually defined
class Anomaly(BaseModel):
    text: str = Field(description="The anomalous text found in the document")
    justification: str = Field(
        description="Brief justification for why this is an anomaly"
    )
    # This field will hold the citation info (e.g., node references)
    source_id: str | None = Field(
        description="Automatically added source reference", default=None
    )


class AnomaliesList(BaseModel):
    anomalies: list[Anomaly] = Field(
        description="List of anomalies found in the document"
    )


# Custom synthesizer that instructs the LLM to extract anomalies in JSON format.
class AnomalyExtractorSynthesizer(BaseSynthesizer):
    def __init__(self, llm=None, nodes=None):
        super().__init__()
        self._llm = llm or Settings.llm
        # Nodes are still provided in case additional context is needed.
        self._nodes = nodes or []

    def _get_prompts(self) -> dict[str, Any]:
        return {}

    def _update_prompts(self, prompts: dict[str, Any]):
        return

    async def aget_response(
        self, query_str: str, text_chunks: list[str], **kwargs: Any
    ) -> RESPONSE_TYPE:
        return self.get_response(query_str, text_chunks, **kwargs)

    def get_response(
        self, query_str: str, text_chunks: list[str], **kwargs: Any
    ) -> str:
        all_text = "\n".join(text_chunks)

        # Prompt must be manually drafted
        # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
        prompt_str = dedent(
            """
        You are an expert document analyzer. Your task is to identify anomalies in the document.
        Anomalies are statements or phrases that seem out of place or inconsistent with the document's context.

        Document:
        {all_text}

        For each anomaly, provide:
        1. The anomalous text (only the specific phrase).
        2. A brief justification for why it is an anomaly.

        Format your answer as a JSON object:
        {{
            "anomalies": [
                {{
                    "text": "anomalous text",
                    "justification": "reason for anomaly",
                }}
            ]
        }}
        """
        )
        print(prompt_str)
        output_parser = PydanticOutputParser(output_cls=AnomaliesList)
        response = self._llm.complete(prompt_str.format(all_text=all_text))

        try:
            parsed_response = output_parser.parse(response.text)
            self._last_anomalies = parsed_response
            return parsed_response.model_dump_json()
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response.text}")
            return "{}"


def extract_anomalies_with_citations(
    document_text: str, api_key: str | None = None
) -> list[Anomaly]:
    """
    Extract anomalies from a document using LlamaIndex with citation support.

    Args:
        document_text: The content of the document.
        api_key: OpenAI API key (if not provided, read from environment variable).

    Returns:
        List of extracted anomalies with automatically added source references.
    """
    openai_api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY")
    llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)
    Settings.llm = llm

    # Create a Document and split it into nodes
    doc = Document(text=document_text)
    splitter = SentenceSplitter(
        paragraph_separator="\n",
        chunk_size=100,
        chunk_overlap=0,
    )
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"Document split into {len(nodes)} nodes")

    # Build a vector index and retriever using all nodes.
    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=len(nodes))

    # Create a custom synthesizer.
    synthesizer = AnomalyExtractorSynthesizer(llm=llm, nodes=nodes)

    # Initialize CitationQueryEngine by passing the expected components.
    citation_query_engine = CitationQueryEngine(
        retriever=retriever,
        llm=llm,
        response_synthesizer=synthesizer,
        citation_chunk_size=100,  # Adjust as needed
        citation_chunk_overlap=10,  # Adjust as needed
    )

    try:
        response = citation_query_engine.query(
            "Extract all anomalies from this document"
        )
        # If the synthesizer stored the anomalies, attach the citation info
        if hasattr(synthesizer, "_last_anomalies"):
            anomalies = synthesizer._last_anomalies.anomalies
            formatted_citations = (
                response.get_formatted_sources()
                if hasattr(response, "get_formatted_sources")
                else None
            )
            for anomaly in anomalies:
                anomaly.source_id = formatted_citations
            return anomalies
        return []

    except Exception as e:
        print(f"Error querying document: {e}")
        return []


# Example usage
document_text = (
    "Consultancy Agreement\n"
    "This agreement between Company A (Supplier) and Company B (Customer)...\n"
    "The term of the agreement is 1 year from the Effective Date...\n"
    "The Supplier shall provide consultancy services as described in Annex 2...\n"
    "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
    "The purple elephant danced gracefully on the moon while eating ice cream.\n"  # anomaly
    "This agreement is governed by the laws of Norway...\n"
)

anomalies = extract_anomalies_with_citations(document_text)
for anomaly in anomalies:
    print(f"Anomaly: {anomaly}")
