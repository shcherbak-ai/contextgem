# ContextGem: Choosing the Tools of an Agent with a Decision Model

import os

from contextgem import (
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    LabelConcept,
    register_tool,
)


# The tools of the agent (here, with made-up results)
@register_tool
def search_contracts(query: str) -> str:
    """Searches the contracts of the company."""
    return "acme-service-agreement"


@register_tool
def look_up_invoice(invoice_number: str) -> str:
    """Looks up an invoice by its number."""
    return '{"amount": "EUR 4,200", "status": "overdue"}'


@register_tool
def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email."""
    return "Sent."


@register_tool
def schedule_meeting(attendees: list[str], start: str) -> str:
    """Schedules a meeting."""
    return "Scheduled."


# The tools in groups, each with a label that says what its tools do
tool_groups = {
    "Search contracts": [search_contracts],
    "Look up invoices": [look_up_invoice],
    "Send emails": [send_email],
    "Schedule meetings": [schedule_meeting],
}

decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",
    api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
)
agent = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
    system_message="You are an assistant of a finance team.",
)

request = "Remind billing@acme.example of invoice 1043, with the amount due."

# Before the turn of the agent: which groups of tools does the request need?
document = Document(raw_text=request)
document.add_concepts(
    [
        LabelConcept(
            name="Tools needed",
            description="Kinds of actions needed to carry out the request",
            labels=list(tool_groups),
            classification_type="multi_label",
            # A tool too many costs less than a tool missing
            decision_threshold=0.3,
        )
    ]
)
document = decision_model.extract_all(document)

# The item has the labels at or above the threshold, each with a probability
# of its own. No item means that no label reached it: keep all the tools then
items = document.get_concept_by_name("Tools needed").extracted_items
groups = items[0].value if items else list(tool_groups)
# groups: ["Look up invoices", "Send emails"]. items[0].probabilities:
#     {"Search contracts": 0.14, "Look up invoices": 0.94, "Send emails": 0.96,
#      "Schedule meetings": 0.02}
agent.tools = [tool for group in groups for tool in tool_groups[group]]

answer = agent.chat(request)
# The agent looked up the invoice, then sent the email.
# answer: "Sent a reminder to billing@acme.example for overdue invoice 1043.
#     The amount due is EUR 4,200."
