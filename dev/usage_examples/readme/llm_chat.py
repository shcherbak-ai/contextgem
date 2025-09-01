# Using LLMs for chat (text + vision), with fallback LLM support

import os

from contextgem import DocumentLLM
from contextgem.public import ChatSession


# Initialize main LLM for chat
main_model = DocumentLLM(
    model="openai/gpt-4o",  # or another provider/model
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),  # your API key for the LLM provider
    system_message="",  # disable default system message for chat, or provide your own
)

# Optional: configure fallback LLM for reliability
fallback_model = DocumentLLM(
    model="openai/gpt-4o-mini",  # or another provider/model
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),  # your API key for the LLM provider
    is_fallback=True,
    system_message="",  # also disable default system message for fallback, or provide your own
)
main_model.fallback_llm = fallback_model


# Preserve conversation history across turns with a ChatSession
session = ChatSession()
first_response = main_model.chat(
    "Hi there!",
    # images=[Image(...)]  # optional: add images for vision models
    chat_session=session,
)
second_response = main_model.chat(
    "And what is EBITDA?",
    chat_session=session,
)
# or use async: `response = await main_model.chat_async(...)`

# Or send a chat message without a session (one-off message → response)
one_off_response = main_model.chat("Test")
