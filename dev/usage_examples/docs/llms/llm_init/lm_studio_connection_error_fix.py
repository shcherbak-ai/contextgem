from contextgem import DocumentLLM


llm = DocumentLLM(
    model="lm_studio/mistralai/mistral-small-3.2",
    api_base="http://localhost:1234/v1",
    api_key="dummy-key",  # dummy key to avoid connection error
)

# This is a known issue with calling LM Studio API in litellm:
# https://github.com/openai/openai-python/issues/961
