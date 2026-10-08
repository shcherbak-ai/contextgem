from contextgem import DocumentLLM


llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
    reasoning_effort="medium",  # Specific to reasoning (CoT-capable) models
    max_tokens=16000,  # Optional; the reasoning tokens count towards this limit
)
