# Example of serializing and deserializing ContextGem document,
# extraction pipeline, and LLM config.

import os
from pathlib import Path

from contextgem import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentLLM,
    DocxConverter,
    ExtractionPipeline,
    StringConcept,
)


# Create a document object
converter = DocxConverter()
docx_path = str(
    Path(__file__).resolve().parents[4]
    / "tests"
    / "docx_files"
    / "en_nda_with_anomalies.docx"
)  # your file path here (Path adapted for testing)
doc = converter.convert(docx_path, strict_mode=True)

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
    model="azure/gpt-4.1-mini",
    api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
    api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
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
