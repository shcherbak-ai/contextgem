from contextgem import DocumentLLM


# Pattern for using any cloud LLM provider
llm = DocumentLLM(
    model="<provider>/<model_name>",
    api_key="<api_key>",
)

# Example - Using OpenAI LLM
llm_openai = DocumentLLM(
    model="openai/gpt-4.1-mini",
    api_key="<api_key>",
    # see DocumentLLM API reference for all configuration options
)

# Example - Using Azure OpenAI LLM
llm_azure_openai = DocumentLLM(
    model="azure/o4-mini",
    api_key="<api_key>",
    api_version="<api_version>",
    api_base="<api_base>",
    # see DocumentLLM API reference for all configuration options
)

# Example - Using OrcaRouter AI gateway
# OrcaRouter is an OpenAI-compatible gateway that can route requests to the best
# model for each task. Prefix any model id with "orcarouter/" (e.g., "orcarouter/auto"
# for automatic routing, "orcarouter/openai/gpt-5.5" to pin a specific model).
# The API key is read from the ORCAROUTER_API_KEY environment variable when not
# provided, and the gateway endpoint defaults to https://api.orcarouter.ai/v1.
# See https://docs.orcarouter.ai for the list of available models.
llm_orcarouter = DocumentLLM(
    model="orcarouter/auto",
    api_key="<api_key>",
    # see DocumentLLM API reference for all configuration options
)
