# LangChain implementation of analyzing multiple documents with a single pipeline,
# with different LLMs, concurrency, and cost tracking
# Jupyter notebook compatible version

import asyncio
import os
import time
from dataclasses import dataclass, field
from textwrap import dedent

import nest_asyncio


nest_asyncio.apply()

from langchain.callbacks import get_openai_callback
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
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
    description: str | None = Field(description="Brief description of the attachment")


class ContractRating(BaseModel):
    """Rating with justification"""

    score: int = Field(description="Rating score (1-10)")
    justification: str = Field(description="Justification for the rating")


class ContractInfo(BaseModel):
    """Complete contract information"""

    contract_type: str = Field(description="Type of contract")
    effective_date: str | None = Field(description="Effective date of the contract")
    governing_law: str | None = Field(description="Governing law of the contract")


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


# Configuration models must be manually defined
@dataclass
class ExtractorConfig:
    """Configuration for a specific extractor"""

    name: str
    description: str
    model_name: str = "gpt-4o-mini"  # Default model


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""

    # Aspect extractors
    party_extractor: ExtractorConfig = field(
        default_factory=lambda: ExtractorConfig(
            name="Contract Parties",
            description="Clauses defining the parties to the agreement",
        )
    )

    term_extractor: ExtractorConfig = field(
        default_factory=lambda: ExtractorConfig(
            name="Term", description="Clauses defining the term of the agreement"
        )
    )

    # Document-level extractors
    contract_info_extractor: ExtractorConfig = field(
        default_factory=lambda: ExtractorConfig(
            name="Contract Information",
            description="Basic contract information including type, date, and governing law",
        )
    )

    attachment_extractor: ExtractorConfig = field(
        default_factory=lambda: ExtractorConfig(
            name="Attachments",
            description="Contract attachments and their descriptions",
        )
    )

    duration_rating_extractor: ExtractorConfig = field(
        default_factory=lambda: ExtractorConfig(
            name="Duration Rating",
            description="Rating of contract duration adequacy",
            model_name="o3-mini",  # Using a more capable model for judgment
        )
    )


# LLM configuration
def get_llm(model_name="gpt-4o-mini", api_key=None):
    """Get a ChatOpenAI instance with the specified configuration"""
    # Skipped temperature etc. for brevity, as e.g. temperature is not supported by o3-mini
    api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY", "")
    return ChatOpenAI(model=model_name, openai_api_key=api_key)


# Chain components must be manually defined
def create_aspect_extractor(aspect_name, aspect_description, model_name="gpt-4o-mini"):
    """Create a chain to extract text related to a specific aspect"""
    llm = get_llm(model_name=model_name)
    parser = PydanticOutputParser(pydantic_object=AspectExtraction)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = PromptTemplate(
        template=dedent(
            """
        You are an expert document analyzer. Extract the text related to the following aspect from the document.
        
        Document:
        {document_text}
        
        Aspect: {aspect_name}
        Description: {aspect_description}
        
        Extract all text related to this aspect.
        {format_instructions}
        """
        ),
        input_variables=["document_text", "aspect_name", "aspect_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )  # this does not provide granular structured content, such as specific paragraphs and sentences

    chain = prompt | llm | parser

    # Return a callable that works with both sync and async code
    def extractor(doc):
        return chain.invoke(
            {
                "document_text": doc,
                "aspect_name": aspect_name,
                "aspect_description": aspect_description,
            }
        )

    # Add an async version that will be used when awaited
    async def async_extractor(doc):
        return await chain.ainvoke(
            {
                "document_text": doc,
                "aspect_name": aspect_name,
                "aspect_description": aspect_description,
            }
        )

    extractor.ainvoke = async_extractor
    return extractor


def create_party_extractor(model_name="gpt-4o-mini"):
    """Create a chain to extract party information"""
    llm = get_llm(model_name=model_name)
    parser = PydanticOutputParser(pydantic_object=PartyExtraction)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = PromptTemplate(
        template=dedent(
            """
        You are an expert document analyzer. Extract all party information from the following contract text.
        
        Contract text:
        {aspect_text}
        
        For each party, extract their name and role in the agreement.
        {format_instructions}
        """
        ),
        input_variables=["aspect_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


def create_term_extractor(model_name="gpt-4o-mini"):
    """Create a chain to extract term information"""
    llm = get_llm(model_name=model_name)
    parser = PydanticOutputParser(pydantic_object=TermExtraction)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = PromptTemplate(
        template=dedent(
            """
        You are an expert document analyzer. Extract term information from the following contract text.
        
        Contract text:
        {aspect_text}
        
        Extract the contract term duration in years. Include the relevant reference text.
        {format_instructions}
        """
        ),
        input_variables=["aspect_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


def create_contract_info_extractor(model_name="gpt-4o-mini"):
    """Create a chain to extract basic contract information"""
    llm = get_llm(model_name=model_name)
    parser = PydanticOutputParser(pydantic_object=ContractInfo)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = PromptTemplate(
        template=dedent(
            """
        You are an expert document analyzer. Extract the following information from the contract document.
        
        Contract document:
        {document_text}
        
        Extract the contract type, effective date if mentioned, and governing law if specified.
        {format_instructions}
        """
        ),
        input_variables=["document_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


def create_attachment_extractor(model_name="gpt-4o-mini"):
    """Create a chain to extract attachment information"""
    llm = get_llm(model_name=model_name)
    parser = PydanticOutputParser(pydantic_object=AttachmentExtraction)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = PromptTemplate(
        template=dedent(
            """
        You are an expert document analyzer. Extract information about all attachments, annexes, 
        schedules, or appendices mentioned in the contract.
        
        Contract document:
        {document_text}
        
        For each attachment, extract:
        1. The title/name of the attachment (e.g., "Appendix A", "Schedule 1", "Annex 2")
        2. A brief description of what the attachment contains (if mentioned in the document)
        
        Example format:
        {{"title": "Appendix A", "description": "Code of conduct"}}
        
        {format_instructions}
        """
        ),
        input_variables=["document_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


def create_duration_rating_extractor(model_name="o3-mini"):
    """Create a chain to rate contract duration adequacy"""
    llm = get_llm(model_name=model_name)
    parser = PydanticOutputParser(pydantic_object=DurationRatingExtraction)

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = PromptTemplate(
        template=dedent(
            """
        You are an expert contract analyst. Evaluate the adequacy of the contract duration 
        considering the subject matter and best practices.
        
        Contract document:
        {document_text}
        
        Rate the duration adequacy on a scale of 1-10, where:
        1 = Extremely inadequate duration
        10 = Perfectly adequate duration
        
        Provide a brief justification for your rating (2-3 sentences).
        {format_instructions}
        """
        ),
        input_variables=["document_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain


# Complete pipeline definition
def create_document_pipeline(config=PipelineConfig()):
    """Create a complete document analysis pipeline and return it along with its components"""

    # Create aspect extractors
    party_aspect_extractor = create_aspect_extractor(
        config.party_extractor.name,
        config.party_extractor.description,
        config.party_extractor.model_name,
    )

    term_aspect_extractor = create_aspect_extractor(
        config.term_extractor.name,
        config.term_extractor.description,
        config.term_extractor.model_name,
    )

    # Create concept extractors for aspects
    party_extractor = create_party_extractor(config.party_extractor.model_name)
    term_extractor = create_term_extractor(config.term_extractor.model_name)

    # Create document-level extractors
    contract_info_extractor = create_contract_info_extractor(
        config.contract_info_extractor.model_name
    )
    attachment_extractor = create_attachment_extractor(
        config.attachment_extractor.model_name
    )
    duration_rating_extractor = create_duration_rating_extractor(
        config.duration_rating_extractor.model_name
    )

    # Create aspect extraction pipeline
    party_pipeline = (
        RunnablePassthrough()
        | party_aspect_extractor
        | RunnableLambda(lambda x: {"aspect_text": x.aspect_text})
        | party_extractor
    )

    term_pipeline = (
        RunnablePassthrough()
        | term_aspect_extractor
        | RunnableLambda(lambda x: {"aspect_text": x.aspect_text})
        | term_extractor
    )

    # Create document-level extraction pipeline
    document_extraction = RunnableParallel(
        contract_info=contract_info_extractor,
        attachments=attachment_extractor,
        duration_rating=duration_rating_extractor,
    )

    # Combine into complete pipeline
    complete_pipeline = RunnableParallel(
        parties=party_pipeline, terms=term_pipeline, document_info=document_extraction
    )

    # Create a components dictionary for easy access
    components = {
        "party_pipeline": party_pipeline,
        "term_pipeline": term_pipeline,
        "contract_info_extractor": contract_info_extractor,
        "attachment_extractor": attachment_extractor,
        "duration_rating_extractor": duration_rating_extractor,
    }

    return complete_pipeline, components


# Cost tracking
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


# Document processing functions
async def process_document_async(
    document_text, pipeline_and_components, cost_tracker=None, use_concurrency=True
):
    """Process a document asynchronously and track costs"""
    pipeline, components = pipeline_and_components  # Unpack the pipeline and components
    results = {}

    # Track tokens used across all calls
    total_tokens = {
        "gpt-4o-mini": {"input": 0, "output": 0},
        "o3-mini": {"input": 0, "output": 0},
    }

    # Use the provided components
    async def process_parties():
        """Process parties using the party pipeline"""
        with get_openai_callback() as cb:
            party_results = await components["party_pipeline"].ainvoke(document_text)
            total_tokens["gpt-4o-mini"]["input"] += cb.prompt_tokens
            total_tokens["gpt-4o-mini"]["output"] += cb.completion_tokens
        return party_results

    async def process_terms():
        """Process terms using the term pipeline"""
        with get_openai_callback() as cb:
            term_results = await components["term_pipeline"].ainvoke(document_text)
            total_tokens["gpt-4o-mini"]["input"] += cb.prompt_tokens
            total_tokens["gpt-4o-mini"]["output"] += cb.completion_tokens
        return term_results

    async def process_contract_info():
        """Process contract info"""
        with get_openai_callback() as cb:
            info_results = await components["contract_info_extractor"].ainvoke(
                document_text
            )
            total_tokens["gpt-4o-mini"]["input"] += cb.prompt_tokens
            total_tokens["gpt-4o-mini"]["output"] += cb.completion_tokens
        return info_results

    async def process_attachments():
        """Process attachments"""
        with get_openai_callback() as cb:
            attachment_results = await components["attachment_extractor"].ainvoke(
                document_text
            )
            total_tokens["gpt-4o-mini"]["input"] += cb.prompt_tokens
            total_tokens["gpt-4o-mini"]["output"] += cb.completion_tokens
        return attachment_results

    async def process_duration_rating():
        """Process duration rating"""
        with get_openai_callback() as cb:
            duration_results = await components["duration_rating_extractor"].ainvoke(
                document_text
            )
            # Duration rating is done with o3-mini
            total_tokens["o3-mini"]["input"] += cb.prompt_tokens
            total_tokens["o3-mini"]["output"] += cb.completion_tokens
        return duration_results

    # Run extractions based on concurrency preference
    if use_concurrency:
        # Process all extractions concurrently for maximum speed
        (
            parties,
            terms,
            contract_info,
            attachments,
            duration_rating,
        ) = await asyncio.gather(
            process_parties(),
            process_terms(),
            process_contract_info(),
            process_attachments(),
            process_duration_rating(),
        )
    else:
        # Process extractions sequentially
        parties = await process_parties()
        terms = await process_terms()
        contract_info = await process_contract_info()
        attachments = await process_attachments()
        duration_rating = await process_duration_rating()

    # Update cost tracker if provided
    if cost_tracker:
        for model, tokens in total_tokens.items():
            cost_tracker.track_usage(model, tokens["input"], tokens["output"])

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


def process_document(
    document_text, pipeline_and_components, cost_tracker=None, use_concurrency=True
):
    """
    Process a document and track costs.
    This is a Jupyter-compatible version that uses the existing event loop
    instead of creating a new one with asyncio.run().
    """
    # Get the current event loop
    loop = asyncio.get_event_loop()
    # Run the async function in the current event loop
    return loop.run_until_complete(
        process_document_async(
            document_text, pipeline_and_components, cost_tracker, use_concurrency
        )
    )


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


# Create cost tracker
cost_tracker = CostTracker()

# Create pipeline with default configuration - returns both pipeline and components
pipeline, pipeline_components = create_document_pipeline()

# Process documents
print("Processing document 1 with concurrency...")
start_time = time.time()
doc1_results = process_document(
    doc1_text, (pipeline, pipeline_components), cost_tracker, use_concurrency=True
)
print(f"Processing time: {time.time() - start_time:.2f} seconds")

print("Processing document 2 with concurrency...")
start_time = time.time()
doc2_results = process_document(
    doc2_text, (pipeline, pipeline_components), cost_tracker, use_concurrency=True
)
print(f"Processing time: {time.time() - start_time:.2f} seconds")

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
