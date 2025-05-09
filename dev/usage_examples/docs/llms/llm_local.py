from contextgem import DocumentLLM

local_llm = DocumentLLM(
    model="ollama/<model_name>",
    api_base="http://localhost:11434",  # Default Ollama endpoint
)

# Example - Using Llama 3.1 LLM via Ollama
llm_llama = DocumentLLM(
    model="ollama/llama3.1:8b",
    api_base="http://localhost:11434",
    # see DocumentLLM API reference for all configuration options
)

# Example - Using DeepSeek R1 reasoning model via Ollama
llm_deepseek = DocumentLLM(
    model="ollama/deepseek-r1:32b",
    api_base="http://localhost:11434",
    # see DocumentLLM API reference for all configuration options
)
