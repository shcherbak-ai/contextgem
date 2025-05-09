from contextgem import DocumentLLM

local_llm = DocumentLLM(
    model="ollama/llama3.1:8b",
    api_base="http://localhost:11434",  # Default Ollama endpoint
)
