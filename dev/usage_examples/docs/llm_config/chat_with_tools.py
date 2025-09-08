import os

from contextgem import ChatSession, DocumentLLM, register_tool


# Define tool handlers and register them
@register_tool
def compute_invoice_total(items: list[dict]) -> str:
    total = 0
    for it in items:
        qty = float(it.get("qty", 0))
        price = float(it.get("price", 0))
        total += qty * price
    return str(total)


# OpenAI-compatible tool schema passed to the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "compute_invoice_total",
            "description": "Compute invoice total as sum(qty*price) over items",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "qty": {"type": "number"},
                                "price": {"type": "number"},
                            },
                            "required": ["qty", "price"],
                        },
                        "minItems": 1,
                    }
                },
                "required": ["items"],
            },
        },
    },
]


# Configure an LLM that supports tool use
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
    system_message="You are a helpful assistant.",  # override default system message for chat
    tools=tools,
)

# Maintain history across turns
session = ChatSession()

prompt = (
    "What's the invoice total for the items "
    "[{'qty':2.0,'price':3.5},{'qty':1.0,'price':3.0}]? "
    "Prices are in USD."
)

answer = llm.chat(prompt, chat_session=session)
print("Answer:", answer)
