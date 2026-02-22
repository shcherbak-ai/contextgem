import os
from typing import TypedDict

from contextgem import ChatSession, DocumentLLM, register_tool


# Define a TypedDict for structured tool parameters.
# This ensures the auto-generated schema includes proper types for each field.
class InvoiceItem(TypedDict):
    qty: float
    price: float


# Define tool handlers with @register_tool decorator.
# The schema is auto-generated from the function signature and docstring.
@register_tool
def compute_invoice_total(items: list[InvoiceItem]) -> str:
    """
    Compute invoice total as sum(qty*price) over items.

    :param items: List of invoice items with quantity and price
    """
    total = 0
    for it in items:
        qty = float(it.get("qty", 0))
        price = float(it.get("price", 0))
        total += qty * price
    return str(total)


# Configure an LLM that supports tool use.
# Simply pass the decorated function(s) directly.
llm = DocumentLLM(
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
    system_message="You are a helpful assistant.",  # override default system message for chat
    tools=[compute_invoice_total],  # Pass the function directly
)

# Maintain history across turns
session = ChatSession()

prompt = (
    "What's the invoice total for the items "
    "[{'qty':2.0,'price':3.5},{'qty':1.0,'price':3.0}]?"
)

answer = llm.chat(prompt, chat_session=session)
print("Answer:", answer)
