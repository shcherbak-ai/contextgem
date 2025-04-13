# Using LLMs for chat (text + vision), with fallback LLM support

import os

from contextgem import DocumentLLM

# from contextgem import Image

main_model = DocumentLLM(
    model="openai/gpt-4o",  # or another provider/model
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),  # your API key for the LLM provider
)

# Optional: fallback LLM
fallback_model = DocumentLLM(
    model="openai/gpt-4o-mini",  # or another provider/model
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),  # your API key for the LLM provider
    is_fallback=True,
)
main_model.fallback_llm = fallback_model

response = main_model.chat(
    "Hello",
    # images=[Image(...)]
)
# or `response = await main_model.chat_async(...)`

print(response)
