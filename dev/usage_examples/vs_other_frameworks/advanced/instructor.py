# Instructor implementation of analyzing multiple documents with a single pipeline,
# with different LLMs, concurrency, and cost tracking
# Jupyter notebook compatible version

import asyncio
import os
from dataclasses import dataclass, field
from textwrap import dedent

import instructor
import nest_asyncio
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field


nest_asyncio.apply()


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


# LLM client setup
def get_client(api_key=None):
    """Get an OpenAI client with instructor integrated"""
    api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)
    return instructor.from_openai(client)


async def get_async_client(api_key=None):
    """Get an AsyncOpenAI client with instructor integrated"""
    api_key = api_key or os.environ.get("CONTEXTGEM_OPENAI_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key)
    return instructor.from_openai(client)


# Helper function to execute completions with token tracking
async def execute_with_tracking(model, messages, response_model, cost_tracker=None):
    """
    Execute a completion request with token tracking.
    """
    # Create the Instructor client
    client = await get_async_client()

    # Make a single API call with Instructor
    response = await client.chat.completions.create(
        model=model, response_model=response_model, messages=messages
    )

    # Access the raw response to get token usage
    if cost_tracker and hasattr(response, "_raw_response"):
        raw_response = response._raw_response
        if hasattr(raw_response, "usage"):
            prompt_tokens = raw_response.usage.prompt_tokens
            completion_tokens = raw_response.usage.completion_tokens
            cost_tracker.track_usage(model, prompt_tokens, completion_tokens)

    return response


def execute_sync(model, messages, response_model):
    """Execute a completion request synchronously"""
    client = get_client()
    return client.chat.completions.create(
        model=model, response_model=response_model, messages=messages
    )


# Unified extraction functions
def extract_aspect(
    document_text,
    aspect_name,
    aspect_description,
    model_name="gpt-4o-mini",
    is_async=False,
    cost_tracker=None,
):
    """Extract text related to a specific aspect"""

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = dedent(
        f"""
    You are an expert document analyzer. Extract the text related to the following aspect from the document.
    
    Document:
    {document_text}
    
    Aspect: {aspect_name}
    Description: {aspect_description}
    
    Extract all text related to this aspect.
    """
    )  # this does not provide granular structured content, such as specific paragraphs and sentences

    messages = [
        {"role": "system", "content": "You are an expert document analyzer."},
        {"role": "user", "content": prompt},
    ]

    if is_async:
        return execute_with_tracking(
            model_name, messages, AspectExtraction, cost_tracker
        )
    else:
        return execute_sync(model_name, messages, AspectExtraction)


def extract_parties(
    aspect_text, model_name="gpt-4o-mini", is_async=False, cost_tracker=None
):
    """Extract party information"""

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = dedent(
        f"""
    You are an expert document analyzer. Extract all party information from the following contract text.
    
    Contract text:
    {aspect_text}
    
    For each party, extract their name and role in the agreement.
    """
    )

    messages = [
        {"role": "system", "content": "You are an expert document analyzer."},
        {"role": "user", "content": prompt},
    ]

    if is_async:
        return execute_with_tracking(
            model_name, messages, PartyExtraction, cost_tracker
        )
    else:
        return execute_sync(model_name, messages, PartyExtraction)


def extract_terms(
    aspect_text, model_name="gpt-4o-mini", is_async=False, cost_tracker=None
):
    """Extract term information"""

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = dedent(
        f"""
    You are an expert document analyzer. Extract term information from the following contract text.
    
    Contract text:
    {aspect_text}
    
    Extract the contract term duration in years. Include the relevant reference text.
    """
    )

    messages = [
        {"role": "system", "content": "You are an expert document analyzer."},
        {"role": "user", "content": prompt},
    ]

    if is_async:
        return execute_with_tracking(model_name, messages, TermExtraction, cost_tracker)
    else:
        return execute_sync(model_name, messages, TermExtraction)


def extract_contract_info(
    document_text, model_name="gpt-4o-mini", is_async=False, cost_tracker=None
):
    """Extract basic contract information"""

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = dedent(
        f"""
    You are an expert document analyzer. Extract the following information from the contract document.
    
    Contract document:
    {document_text}
    
    Extract the contract type, effective date if mentioned, and governing law if specified.
    """
    )

    messages = [
        {"role": "system", "content": "You are an expert document analyzer."},
        {"role": "user", "content": prompt},
    ]

    if is_async:
        return execute_with_tracking(model_name, messages, ContractInfo, cost_tracker)
    else:
        return execute_sync(model_name, messages, ContractInfo)


def extract_attachments(
    document_text, model_name="gpt-4o-mini", is_async=False, cost_tracker=None
):
    """Extract attachment information"""

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = dedent(
        f"""
    You are an expert document analyzer. Extract information about all attachments, annexes, 
    schedules, or appendices mentioned in the contract.
    
    Contract document:
    {document_text}
    
    For each attachment, extract:
    1. The title/name of the attachment (e.g., "Appendix A", "Schedule 1", "Annex 2")
    2. A brief description of what the attachment contains (if mentioned in the document)
    """
    )

    messages = [
        {"role": "system", "content": "You are an expert document analyzer."},
        {"role": "user", "content": prompt},
    ]

    if is_async:
        return execute_with_tracking(
            model_name, messages, AttachmentExtraction, cost_tracker
        )
    else:
        return execute_sync(model_name, messages, AttachmentExtraction)


def extract_duration_rating(
    document_text, model_name="o3-mini", is_async=False, cost_tracker=None
):
    """Rate contract duration adequacy"""

    # Prompt must be manually drafted
    # This is a basic example, which is shortened for brevity. The prompt should be improved for better accuracy.
    prompt = dedent(
        f"""
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

    messages = [
        {"role": "system", "content": "You are an expert contract analyst."},
        {"role": "user", "content": prompt},
    ]

    if is_async:
        return execute_with_tracking(
            model_name, messages, DurationRatingExtraction, cost_tracker
        )
    else:
        return execute_sync(model_name, messages, DurationRatingExtraction)


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
    document_text, config=None, cost_tracker=None, use_concurrency=True
):
    """Process a document asynchronously and track costs"""
    if config is None:
        config = PipelineConfig()

    results = {}

    # Define processing functions
    async def process_party_pipeline():
        # Extract party aspect
        party_aspect = await extract_aspect(
            document_text,
            config.party_extractor.name,
            config.party_extractor.description,
            model_name=config.party_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

        # Extract parties from the aspect
        parties = await extract_parties(
            party_aspect.aspect_text,
            model_name=config.party_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

        return parties

    async def process_term_pipeline():
        # Extract term aspect
        term_aspect = await extract_aspect(
            document_text,
            config.term_extractor.name,
            config.term_extractor.description,
            model_name=config.term_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

        # Extract terms from the aspect
        terms = await extract_terms(
            term_aspect.aspect_text,
            model_name=config.term_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

        return terms

    async def process_contract_info():
        return await extract_contract_info(
            document_text,
            model_name=config.contract_info_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

    async def process_attachments():
        return await extract_attachments(
            document_text,
            model_name=config.attachment_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

    async def process_duration_rating():
        return await extract_duration_rating(
            document_text,
            model_name=config.duration_rating_extractor.model_name,
            is_async=True,
            cost_tracker=cost_tracker,
        )

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
            process_party_pipeline(),
            process_term_pipeline(),
            process_contract_info(),
            process_attachments(),
            process_duration_rating(),
        )
    else:
        # Process extractions sequentially
        parties = await process_party_pipeline()
        terms = await process_term_pipeline()
        contract_info = await process_contract_info()
        attachments = await process_attachments()
        duration_rating = await process_duration_rating()

    # Structure results in the same format as the LangChain implementation
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
    document_text, config=None, cost_tracker=None, use_concurrency=True
):
    """
    Process a document and track costs.
    """
    # Get the current event loop
    loop = asyncio.get_event_loop()
    # Run the async function in the current event loop
    return loop.run_until_complete(
        process_document_async(document_text, config, cost_tracker, use_concurrency)
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

# Create pipeline with default configuration
config = PipelineConfig()

# Process documents
print("Processing document 1 with concurrency...")
doc1_results = process_document(doc1_text, config, cost_tracker, use_concurrency=True)

print("Processing document 2 with concurrency...")
doc2_results = process_document(doc2_text, config, cost_tracker, use_concurrency=True)

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
