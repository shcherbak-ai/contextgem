from contextgem import DocumentLLM


local_llm = DocumentLLM(
    model="ollama_chat/<model_name>",
    api_base="http://localhost:11434",  # Default Ollama endpoint
)

# Example - Using Llama 3.3 LLM via Ollama
llm_llama = DocumentLLM(
    model="ollama_chat/llama3.3:70b",
    api_base="http://localhost:11434",
    # see DocumentLLM API reference for all configuration options
)

# Example - Using DeepSeek R1 reasoning model via Ollama
llm_deepseek = DocumentLLM(
    model="ollama_chat/deepseek-r1:32b",
    api_base="http://localhost:11434",
    # see DocumentLLM API reference for all configuration options
)

# Example - Using Mistral Small 3.2 via LM Studio (no API key is needed)
llm_mistral = DocumentLLM(
    model="lm_studio/mistralai/mistral-small-3.2",
    api_base="http://localhost:1234/v1",  # Default LM Studio endpoint
    # see DocumentLLM API reference for all configuration options
)
