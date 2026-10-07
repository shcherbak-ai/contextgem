# Example of serializing and deserializing ContextGem document,
# extraction pipeline, and LLM config.

import os
from pathlib import Path

from contextgem import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentLLM,
    ExtractionPipeline,
    StringConcept,
)


# Create a document object
text_path = (
    Path(__file__).resolve().parents[4] / "tests" / "ndas" / "en_nda_with_anomalies.txt"
)  # your file path here (Path adapted for testing)
doc = Document(raw_text=text_path.read_text(encoding="utf-8"))

# Create an extraction pipeline
extraction_pipeline = ExtractionPipeline(
    aspects=[
        Aspect(
            name="Categories of confidential information",
            description="Clauses describing confidential information covered by the NDA",
            concepts=[
                StringConcept(
                    name="Types of disclosure",
                    description="Types of disclosure of confidential information",
                ),
                # ...
            ],
        ),
        # ...
    ],
    concepts=[
        BooleanConcept(
            name="Is mutual",
            description="Whether the NDA is mutual (both parties act as discloser/recipient)",
            add_justifications=True,
        ),
        # ...
    ],
)

# Attach the pipeline to the document
doc.assign_pipeline(extraction_pipeline)

# Configure a document LLM with your API parameters
llm = DocumentLLM(
    model="openai/gpt-6-luna",
    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
)

# Extract data from the document
doc = llm.extract_all(doc)

# Serialize the LLM config, pipeline and document
llm_config_json = llm.to_json()  # or to_dict() / to_disk()
extraction_pipeline_json = extraction_pipeline.to_json()  # or to_dict() / to_disk()
processed_doc_json = doc.to_json()  # or to_dict() / to_disk()

# Deserialize the LLM config, pipeline and document
llm_deserialized = DocumentLLM.from_json(
    llm_config_json
)  # or from_dict() / from_disk()
extraction_pipeline_deserialized = ExtractionPipeline.from_json(
    extraction_pipeline_json
)  # or from_dict() / from_disk()
processed_doc_deserialized = Document.from_json(
    processed_doc_json
)  # or from_dict() / from_disk()

# All extracted data is preserved!
assert processed_doc_deserialized.aspects[0].concepts[0].extracted_items
