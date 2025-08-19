from contextgem import DocumentLLM


llm = DocumentLLM(
    model="openai/o3-mini",
    api_key="<your-openai-api-key>",
    max_completion_tokens=8000,  # Specific to reasoning (CoT-capable) models
    reasoning_effort="medium",  # Optional
)
