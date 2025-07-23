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
