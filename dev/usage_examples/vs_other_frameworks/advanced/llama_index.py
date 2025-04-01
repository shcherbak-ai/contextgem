# LlamaIndex implementation of analyzing multiple documents with a single pipeline,
# with different LLMs, concurrency, and cost tracking
# Jupyter notebook compatible version

import asyncio
import os
from textwrap import dedent
from typing import Optional

import nest_asyncio

nest_asyncio.apply()

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field


# Pydantic models must be manually defined
class PartyInfo(BaseModel):
    """Information about contract parties"""

    name: str = Field(description="Name of the party")
    role: str = Field(description="Role of the party (e.g., Client, Provider)")


class Term(BaseModel):
    """Contract term information"""

    duration_years: int = Field(description="Duration in years")
    reference: str = Field(
        description="Reference text from document"
    )  # LLM reciting a reference is error-prone and unreliable


class Attachment(BaseModel):
    """Contract attachment information"""

    title: str = Field(description="Title of the attachment")
    description: Optional[str] = Field(
        description="Brief description of the attachment"
    )


class ContractRating(BaseModel):
    """Rating with justification"""

    score: int = Field(description="Rating score (1-10)")
    justification: str = Field(description="Justification for the rating")


class ContractInfo(BaseModel):
    """Complete contract information"""

    contract_type: str = Field(description="Type of contract")
    effective_date: Optional[str] = Field(description="Effective date of the contract")
    governing_law: Optional[str] = Field(description="Governing law of the contract")


class AspectExtraction(BaseModel):
    """Result of aspect extraction"""

    aspect_text: str = Field(
        description="Extracted text for this aspect"
    )  # this does not provide granular structured content, such as specific paragraphs and sentences


class PartyExtraction(BaseModel):
    """Party extraction results"""

    parties: list[PartyInfo] = Field(description="List of parties in the contract")


class TermExtraction(BaseModel):
    """Term extraction results"""

    terms: list[Term] = Field(description="Contract term details")


class AttachmentExtraction(BaseModel):
    """Attachment extraction results"""

    attachments: list[Attachment] = Field(description="List of contract attachments")


class DurationRatingExtraction(BaseModel):
    """Duration adequacy rating"""

    rating: ContractRating = Field(description="Rating of contract duration adequacy")


# Cost tracking class
class CostTracker:
    """Track LLM costs across multiple extractions"""

    def __init__(self):
        self.costs = {
            "gpt-4o-mini": {
                "input_per_1m": 0.15,
                "output_per_1m": 0.60,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "o3-mini": {
                "input_per_1m": 1.10,
                "output_per_1m": 4.40,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        }
        self.total_cost = 0.0

    def track_usage(self, model_name, input_tokens, output_tokens):
        """Track token usage for a model"""
        # Extract base model name
        base_model = model_name.split("/")[-1] if "/" in model_name else model_name

        if base_model in self.costs:
            self.costs[base_model]["input_tokens"] += input_tokens
            self.costs[base_model]["output_tokens"] += output_tokens

            # Calculate costs separately for input and output tokens
            input_cost = input_tokens * (
                self.costs[base_model]["input_per_1m"] / 1000000
            )
            output_cost = output_tokens * (
                self.costs[base_model]["output_per_1m"] / 1000000
            )

            self.total_cost += input_cost + output_cost

    def get_costs(self):
        """Get cost summary"""
        model_costs = {}
        for model, data in self.costs.items():
            if data["input_tokens"] > 0 or data["output_tokens"] > 0:
                input_cost = data["input_tokens"] * (data["input_per_1m"] / 1000000)
                output_cost = data["output_tokens"] * (data["output_per_1m"] / 1000000)
                model_costs[model] = {
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": input_cost + output_cost,
                    "input_tokens": data["input_tokens"],
                    "output_tokens": data["output_tokens"],
                }

        return {
            "model_costs": model_costs,
            "total_cost": self.total_cost,
        }


# Helper functions for extractors
def get_llm(model_name="gpt-4o-mini", api_key=None, temperature=0, token_counter=None):
    """Get an OpenAI instance with the specified configuration"""
    api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY", "")

    # Create callback manager with token counter if provided
    callback_manager = None
    if token_counter is not None:
        callback_manager = CallbackManager([token_counter])

    return OpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        callback_manager=callback_manager,
    )


def create_aspect_extractor(
    aspect_name, aspect_description, model_name="gpt-4o-mini", token_counter=None
):
    """Create an extractor to extract text related to a specific aspect"""
    llm = get_llm(model_name=model_name, token_counter=token_counter)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        f"""
    You are an expert document analyzer. Extract the text related to the following aspect from the document.
    
    Document:
    {{document_text}}
    
    Aspect: {aspect_name}
    Description: {aspect_description}
    
    Extract all text related to this aspect.
    """
    )  # this does not provide granular structured content, such as specific paragraphs and sentences

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=AspectExtraction),
        prompt_template_str=prompt_template,
        llm=llm,
    )
    return program


def create_party_extractor(model_name="gpt-4o-mini", token_counter=None):
    """Create an extractor for party information"""
    llm = get_llm(model_name=model_name, token_counter=token_counter)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        """
    You are an expert document analyzer. Extract all party information from the following contract text.
    
    Contract text:
    {aspect_text}
    
    For each party, extract their name and role in the agreement.
    """
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=PartyExtraction),
        prompt_template_str=prompt_template,
        llm=llm,
    )
    return program


def create_term_extractor(model_name="gpt-4o-mini", token_counter=None):
    """Create an extractor for term information"""
    llm = get_llm(model_name=model_name, token_counter=token_counter)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        """
    You are an expert document analyzer. Extract term information from the following contract text.
    
    Contract text:
    {aspect_text}
    
    Extract the contract term duration in years. Include the relevant reference text.
    """
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=TermExtraction),
        prompt_template_str=prompt_template,
        llm=llm,
    )
    return program


def create_contract_info_extractor(model_name="gpt-4o-mini", token_counter=None):
    """Create an extractor for basic contract information"""
    llm = get_llm(model_name=model_name, token_counter=token_counter)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        """
    You are an expert document analyzer. Extract the following information from the contract document.
    
    Contract document:
    {document_text}
    
    Extract the contract type, effective date if mentioned, and governing law if specified.
    """
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=ContractInfo),
        prompt_template_str=prompt_template,
        llm=llm,
    )
    return program


def create_attachment_extractor(model_name="gpt-4o-mini", token_counter=None):
    """Create an extractor for attachment information"""
    llm = get_llm(model_name=model_name, token_counter=token_counter)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        """
    You are an expert document analyzer. Extract information about all attachments, annexes, 
    schedules, or appendices mentioned in the contract.
    
    Contract document:
    {document_text}
    
    For each attachment, extract:
    1. The title/name of the attachment (e.g., "Appendix A", "Schedule 1", "Annex 2")
    2. A brief description of what the attachment contains (if mentioned in the document)
    
    Example format:
    {"title": "Appendix A", "description": "Code of conduct"}
    """
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=AttachmentExtraction),
        prompt_template_str=prompt_template,
        llm=llm,
    )
    return program


def create_duration_rating_extractor(model_name="o3-mini", token_counter=None):
    """Create an extractor to rate contract duration adequacy"""
    llm = get_llm(model_name=model_name, token_counter=token_counter)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt_template = dedent(
        """
    You are an expert contract analyst. Evaluate the adequacy of the contract duration 
    considering the subject matter and best practices.
    
    Contract document:
    {document_text}
    
    Rate the duration adequacy on a scale of 1-10, where:
    1 = Extremely inadequate duration
    10 = Perfectly adequate duration
    
    Provide a brief justification for your rating (2-3 sentences).
    """
    )

    program = LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=DurationRatingExtraction),
        prompt_template_str=prompt_template,
        llm=llm,
    )
    return program


# Main document processing functions
async def process_document_async(
    document_text, cost_tracker=None, use_concurrency=True
):
    """Process a document asynchronously and track costs"""
    results = {}

    # Create separate token counting handlers for each model
    gpt4o_token_counter = TokenCountingHandler()
    o3_token_counter = TokenCountingHandler()

    # Create extractors with appropriate token counters
    party_aspect_extractor = create_aspect_extractor(
        "Contract Parties",
        "Clauses defining the parties to the agreement",
        token_counter=gpt4o_token_counter,
    )
    term_aspect_extractor = create_aspect_extractor(
        "Term",
        "Clauses defining the term of the agreement",
        token_counter=gpt4o_token_counter,
    )
    party_extractor = create_party_extractor(token_counter=gpt4o_token_counter)
    term_extractor = create_term_extractor(token_counter=gpt4o_token_counter)
    contract_info_extractor = create_contract_info_extractor(
        token_counter=gpt4o_token_counter
    )
    attachment_extractor = create_attachment_extractor(
        token_counter=gpt4o_token_counter
    )

    # Use separate token counter for o3-mini
    duration_rating_extractor = create_duration_rating_extractor(
        model_name="o3-mini", token_counter=o3_token_counter
    )

    # Define processing functions using native async methods
    async def process_party_aspect():
        response = await party_aspect_extractor.acall(document_text=document_text)
        return response

    async def process_term_aspect():
        response = await term_aspect_extractor.acall(document_text=document_text)
        return response

    # Get aspect texts
    if use_concurrency:
        party_aspect, term_aspect = await asyncio.gather(
            process_party_aspect(), process_term_aspect()
        )
    else:
        party_aspect = await process_party_aspect()
        term_aspect = await process_term_aspect()

    async def process_parties():
        party_results = await party_extractor.acall(
            aspect_text=party_aspect.aspect_text
        )
        return party_results

    async def process_terms():
        term_results = await term_extractor.acall(aspect_text=term_aspect.aspect_text)
        return term_results

    async def process_contract_info():
        contract_info = await contract_info_extractor.acall(document_text=document_text)
        return contract_info

    async def process_attachments():
        attachments = await attachment_extractor.acall(document_text=document_text)
        return attachments

    async def process_duration_rating():
        duration_rating = await duration_rating_extractor.acall(
            document_text=document_text
        )
        return duration_rating

    # Run extractions based on concurrency preference
    if use_concurrency:
        parties, terms, contract_info, attachments, duration_rating = (
            await asyncio.gather(
                process_parties(),
                process_terms(),
                process_contract_info(),
                process_attachments(),
                process_duration_rating(),
            )
        )
    else:
        parties = await process_parties()
        terms = await process_terms()
        contract_info = await process_contract_info()
        attachments = await process_attachments()
        duration_rating = await process_duration_rating()

    # Get token usage from the token counter and update cost tracker
    if cost_tracker:
        cost_tracker.track_usage(
            "gpt-4o-mini",
            gpt4o_token_counter.prompt_llm_token_count,
            gpt4o_token_counter.completion_llm_token_count,
        )
        cost_tracker.track_usage(
            "o3-mini",
            o3_token_counter.prompt_llm_token_count,
            o3_token_counter.completion_llm_token_count,
        )

    # Structure results in an easy-to-use format
    results["contract_type"] = contract_info.contract_type
    results["governing_law"] = contract_info.governing_law
    results["effective_date"] = contract_info.effective_date
    results["parties"] = parties.parties
    results["term_years"] = terms.terms[0].duration_years if terms.terms else None
    results["term_reference"] = terms.terms[0].reference if terms.terms else None
    results["attachments"] = attachments.attachments
    results["duration_rating"] = duration_rating.rating

    return results


def process_document(document_text, cost_tracker=None, use_concurrency=True):
    """
    Process a document and track costs.
    This is a Jupyter-compatible version that uses the existing event loop
    instead of creating a new one with asyncio.run().
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        process_document_async(document_text, cost_tracker, use_concurrency)
    )


# Function to pretty-print document results
def print_document_results(doc_name, results):
    print(f"\nResults from {doc_name}:")
    print(f"Contract Type: {results['contract_type']}")
    print(f"Parties: {[f'{p.name} ({p.role})' for p in results['parties']]}")
    print(f"Term: {results['term_years']} years")
    print(
        f"Term Reference: {results['term_reference'] if results['term_reference'] else 'Not specified'}"
    )
    print(f"Governing Law: {results['governing_law']}")
    print(f"Attachments: {[(a.title, a.description) for a in results['attachments']]}")
    print(f"Duration Rating: {results['duration_rating'].score}/10")
    print(f"Rating Justification: {results['duration_rating'].justification}")


# Example usage
# Sample contract texts (shortened for brevity)
doc1_text = (
    "Consultancy Agreement\n"
    "This agreement between Company A (Supplier) and Company B (Customer)...\n"
    "The term of the agreement is 1 year from the Effective Date...\n"
    "The Supplier shall provide consultancy services as described in Annex 2...\n"
    "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
    "All intellectual property created during the provision of services shall belong to the Customer...\n"
    "This agreement is governed by the laws of Norway...\n"
    "Annex 1: Data processing agreement...\n"
    "Annex 2: Statement of Work...\n"
    "Annex 3: Service Level Agreement...\n"
)

doc2_text = (
    "Service Level Agreement\n"
    "This agreement between TechCorp (Provider) and GlobalInc (Client)...\n"
    "The agreement shall commence on January 1, 2023 and continue for 2 years...\n"
    "The Provider shall deliver IT support services as outlined in Schedule A...\n"
    "The Client shall make monthly payments of $5,000 within 15 days of invoice receipt...\n"
    "The Provider guarantees [99.9%] uptime for all critical systems...\n"
    "Either party may terminate with 60 days written notice...\n"
    "This agreement is governed by the laws of California...\n"
    "Schedule A: Service Descriptions...\n"
    "Schedule B: Response Time Requirements...\n"
)


# Create cost tracker
cost_tracker = CostTracker()

# Process documents
print("Processing document 1 with concurrency...")
doc1_results = process_document(doc1_text, cost_tracker, use_concurrency=True)

print("Processing document 2 with concurrency...")
doc2_results = process_document(doc2_text, cost_tracker, use_concurrency=True)

# Print results
print_document_results("Document 1 (Consultancy Agreement)", doc1_results)
print_document_results("Document 2 (Service Level Agreement)", doc2_results)

# Print cost information
print("\nProcessing costs:")
costs = cost_tracker.get_costs()
for model, model_data in costs["model_costs"].items():
    print(f"\n{model}:")
    print(f"  Input cost: ${model_data['input_cost']:.4f}")
    print(f"  Output cost: ${model_data['output_cost']:.4f}")
    print(f"  Total cost: ${model_data['total_cost']:.4f}")
print(f"\nTotal across all models: ${costs['total_cost']:.4f}")
