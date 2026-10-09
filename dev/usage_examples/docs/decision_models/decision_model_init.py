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

# Cloudflare: the API base has the ID of your account
decision_model_cloudflare = DocumentDecisionModel(
    model="cloudflare/clef",
    api_key="<your-cloudflare-api-key>",
    api_base="https://api.cloudflare.com/client/v4/accounts/<your-account-id>/ai/run",
)

# Self-hosted server that serves `POST <api_base>/v1/systemone`. No API key is needed.
decision_model_self_hosted = DocumentDecisionModel(
    model="strands_decider/strands-decider-2B-hobson-v19",  # or the model of your server
    api_base="http://localhost:8000",
)
