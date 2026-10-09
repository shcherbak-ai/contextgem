from contextgem import DocumentDecisionModel


local_decision_model = DocumentDecisionModel(
    model="strands_decider/<model_name>",  # the prefix of a self-hosted server
    api_base="http://localhost:11434",  # Default Ollama endpoint
)

# Example - Using the Nimble decision model via Ollama (`ollama pull nimble`)
decision_model_ollama = DocumentDecisionModel(
    model="strands_decider/nimble",
    api_base="http://localhost:11434",
    # see DocumentDecisionModel API reference for all configuration options
)

# Example - Using the Kev 4B decision model via LM Studio
# (`lms get https://huggingface.co/ggml-org/Kev-4B-GGUF`)
decision_model_lm_studio = DocumentDecisionModel(
    model="strands_decider/kev-4b",
    api_base="http://localhost:1234",  # Default LM Studio endpoint
)
