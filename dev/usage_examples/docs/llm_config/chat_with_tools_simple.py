import os

from contextgem import ChatSession, DocumentLLM, register_tool


@register_tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.

    :param city: Name of the city
    """
    # In a real app, this would call a weather API
    return f"Weather in {city}: Sunny, 22°C"


llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
    system_message="You are a helpful assistant.",
    tools=[get_weather],
)

session = ChatSession()
answer = llm.chat("What's the weather in Tokyo?", chat_session=session)
print("Answer:", answer)
