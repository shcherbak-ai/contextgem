from contextgem import DocumentDecisionModel


# TypeSafe
decision_model_typesafe = DocumentDecisionModel(
    model="typesafe/jev-latest",  # Format: <provider>/<model_name>
    api_key="<your-typesafe-api-key>",
)

# OpenAI (Decisions API)
decision_model_openai = DocumentDecisionModel(
    model="openai/gpt-6-luna",
    api_key="<your-openai-api-key>",
)

# Microsoft Foundry: the name after the prefix is the name of your deployment,
# and the API base is the endpoint of your Foundry resource
decision_model_foundry = DocumentDecisionModel(
    model="azure_ai/<your-deployment-name>",
    api_key="<your-foundry-api-key>",
    api_base="https://<your-resource>.services.ai.azure.com",
)

# Cloudflare: the API base has the ID of your account
decision_model_cloudflare = DocumentDecisionModel(
    model="cloudflare/clef",
    api_key="<your-cloudflare-api-key>",
    api_base="https://api.cloudflare.com/client/v4/accounts/<your-account-id>/ai/run",
)

# Local or self-hosted server that serves `POST <api_base>/v1/systemone`,
# e.g. Ollama. No API key is needed.
decision_model_local = DocumentDecisionModel(
    model="strands_decider/nimble",  # the model of your server
    api_base="http://localhost:11434",  # Default Ollama endpoint
)
