from contextgem import DocumentLLM

local_llm = DocumentLLM(
    model="ollama_chat/llama3.3:70b",
    api_base="http://localhost:11434",  # Default Ollama endpoint
)
