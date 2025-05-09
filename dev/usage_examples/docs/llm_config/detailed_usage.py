from contextgem import DocumentLLM

llm = DocumentLLM(
    model="openai/gpt-4.1",
    api_key="<your-openai-api-key>",
)

# Perform some extraction tasks

usage_info = llm.get_usage()

# Access the first usage container in the list (for the primary LLM)
llm_usage = usage_info[0]

# Get detailed call information
for call in llm_usage.usage.calls:
    print(f"Prompt: {call.prompt}")
    print(f"Response: {call.response}")  # original, unprocessed response
    print(f"Sent at: {call.timestamp_sent}")
    print(f"Received at: {call.timestamp_received}")
