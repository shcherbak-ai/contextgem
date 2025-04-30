![ContextGem](https://contextgem.dev/_static/contextgem_readme_header.png "ContextGem - Effortless LLM extraction from documents")

# ContextGem: Effortless LLM extraction from documents

[![tests](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SergiiShcherbak/daaee00e1dfff7a29ca10a922ec3becd/raw/coverage.json)](https://github.com/shcherbak-ai/contextgem/actions)
[![docs](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml)
[![documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://shcherbak-ai.github.io/contextgem/)
[![License](https://img.shields.io/badge/License-Apache_2.0-bright.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI](https://img.shields.io/pypi/v/contextgem)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![Code Security](https://github.com/shcherbak-ai/contextgem/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

<img src="https://contextgem.dev/_static/tab_solid.png" alt="ContextGem: 2nd Product of the week" width="250">
<br/><br/>

ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code.


## 💎 Why ContextGem?

Most popular LLM frameworks for extracting structured data from documents require extensive boilerplate code to extract even basic information. This significantly increases development time and complexity.

ContextGem addresses this challenge by providing a flexible, intuitive framework that extracts structured data and insights from documents with minimal effort. Complex, most time-consuming parts are handled with **powerful abstractions**, eliminating boilerplate code and reducing development overhead.

Read more on the project [motivation](https://contextgem.dev/motivation.html) in the documentation.


## ⭐ Key features

<table>
    <thead>
        <tr style="text-align: left; opacity: 0.8;">
            <th style="width: 75%">Built-in abstractions</th>
            <th style="width: 10%"><strong>ContextGem</strong></th>
            <th style="width: 15%">Other LLM frameworks*</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                Automated dynamic prompts
            </td>
            <td>🟢</td>
            <td>◯</td>
        </tr>
        <tr>
            <td>
                Automated data modelling and validators
            </td>
            <td>🟢</td>
            <td>◯</td>
        </tr>
        <tr>
            <td>
                Precise granular reference mapping (paragraphs & sentences)
            </td>
            <td>🟢</td>
            <td>◯</td>
        </tr>
        <tr>
            <td>
                Justifications (reasoning backing the extraction)
            </td>
            <td>🟢</td>
            <td>◯</td>
        </tr>
        <tr>
            <td>
                Neural segmentation (SaT)
            </td>
            <td>🟢</td>
            <td>◯</td>
        </tr>
        <tr>
            <td>
                Multilingual support (I/O without prompting)
            </td>
            <td>🟢</td>
            <td>◯</td>
        </tr>
        <tr>
            <td>
                Single, unified extraction pipeline (declarative, reusable, fully serializable)
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Grouped LLMs with role-specific tasks
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Nested context extraction
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Unified, fully serializable results storage model (document)
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Extraction task calibration with examples
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Built-in concurrent I/O processing
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Automated usage & costs tracking
            </td>
            <td>🟢</td>
            <td>🟡</td>
        </tr>
        <tr>
            <td>
                Fallback and retry logic
            </td>
            <td>🟢</td>
            <td>🟢</td>
        </tr>
        <tr>
            <td>
                Multiple LLM providers
            </td>
            <td>🟢</td>
            <td>🟢</td>
        </tr>
    </tbody>
</table>

🟢 - fully supported - no additional setup required<br>
🟡 - partially supported - requires additional setup<br>
◯ - not supported - requires custom logic

\* See [descriptions](https://contextgem.dev/motivation.html#the-contextgem-solution) of ContextGem abstractions and [comparisons](https://contextgem.dev/vs_other_frameworks.html) of specific implementation examples using ContextGem and other popular open-source LLM frameworks.


## 💡 With **minimal code**, you can:

- **Extract structured data** from documents (text, images)
- **Identify and analyze key aspects** (topics, themes, categories) within documents
- **Extract specific concepts** (entities, facts, conclusions, assessments) from documents
- **Build complex extraction workflows** through a simple, intuitive API
- **Create multi-level extraction pipelines** (aspects containing concepts, hierarchical aspects)

<br/>

![ContextGem extraction example](https://contextgem.dev/_static/readme_code_snippet.png "ContextGem extraction example")


## 📦 Installation

```bash
pip install -U contextgem
```


## 🚀 Quick start

```python
# Quick Start Example - Extracting anomalies from a document, with source references and justifications

import os

from contextgem import Document, DocumentLLM, StringConcept

# Sample document text (shortened for brevity)
doc = Document(
    raw_text=(
        "Consultancy Agreement\n"
        "This agreement between Company A (Supplier) and Company B (Customer)...\n"
        "The term of the agreement is 1 year from the Effective Date...\n"
        "The Supplier shall provide consultancy services as described in Annex 2...\n"
        "The Customer shall pay the Supplier within 30 calendar days of receiving an invoice...\n"
        "The purple elephant danced gracefully on the moon while eating ice cream.\n"  # 💎 anomaly
        "This agreement is governed by the laws of Norway...\n"
    ),
)

# Attach a document-level concept
doc.concepts = [
    StringConcept(
        name="Anomalies",  # in longer contexts, this concept is hard to capture with RAG
        description="Anomalies in the document",
        add_references=True,
        reference_depth="sentences",
        add_justifications=True,
        justification_depth="brief",
        # see the docs for more configuration options
    )
    # add more concepts to the document, if needed
    # see the docs for available concepts: StringConcept, JsonObjectConcept, etc.
]
# Or use `doc.add_concepts([...])`

# Define an LLM for extracting information from the document
llm = DocumentLLM(
    model="openai/gpt-4o-mini",  # or another provider/LLM
    api_key=os.environ.get(
        "CONTEXTGEM_OPENAI_API_KEY"
    ),  # your API key for the LLM provider
    # see the docs for more configuration options
)

# Extract information from the document
doc = llm.extract_all(doc)  # or use async version `await llm.extract_all_async(doc)`

# Access extracted information in the document object
print(
    doc.concepts[0].extracted_items
)  # extracted items with references & justifications
# or `doc.get_concept_by_name("Anomalies").extracted_items`

```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/readme/quickstart_concept.ipynb)

---

See more examples in the documentation:

### Basic usage examples
- [Aspect Extraction from Document](https://contextgem.dev/quickstart.html#aspect-extraction-from-document)
- [Extracting Aspect with Sub-Aspects](https://contextgem.dev/quickstart.html#extracting-aspect-with-sub-aspects)
- [Concept Extraction from Aspect](https://contextgem.dev/quickstart.html#concept-extraction-from-aspect)
- [Concept Extraction from Document (text)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-text)
- [Concept Extraction from Document (vision)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-vision)
- [LLM chat interface](https://contextgem.dev/quickstart.html#lightweight-llm-chat-interface)

### Advanced usage examples
- [Extracting Aspects Containing Concepts](https://contextgem.dev/advanced_usage.html#extracting-aspects-with-concepts)
- [Extracting Aspects and Concepts from a Document](https://contextgem.dev/advanced_usage.html#extracting-aspects-and-concepts-from-a-document)
- [Using a Multi-LLM Pipeline to Extract Data from Several Documents](https://contextgem.dev/advanced_usage.html#using-a-multi-llm-pipeline-to-extract-data-from-several-documents)


## 🔄 Document converters

To create a ContextGem document for LLM analysis, you can either pass raw text directly, or use built-in converters that handle various file formats.

### 📄 DOCX converter

ContextGem provides built-in converter to easily transform DOCX files into LLM-ready data.

- Extracts information that other open-source tools often do not capture: misaligned tables, comments, footnotes, textboxes, headers/footers, and embedded images
- Preserves document structure with rich metadata for improved LLM analysis

```python
# Using ContextGem's DocxConverter

from contextgem import DocxConverter

converter = DocxConverter()

# Convert a DOCX file to an LLM-ready ContextGem Document
# from path
document = converter.convert("path/to/document.docx")
# or from file object
with open("path/to/document.docx", "rb") as docx_file_object:
    document = converter.convert(docx_file_object)

# You can also use it as a standalone text extractor
docx_text = converter.convert_to_text_format(
    "path/to/document.docx",
    output_format="markdown",  # or "raw"
)

```

Learn more about [DOCX converter features](https://contextgem.dev/converters/docx.html) in the documentation.


## 🎯 Focused document analysis

ContextGem leverages LLMs' long context windows to deliver superior extraction accuracy from individual documents. Unlike RAG approaches that often [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f), ContextGem capitalizes on [continuously expanding context capacity](https://arxiv.org/abs/2502.12962), evolving LLM capabilities, and decreasing costs. This focused approach enables direct information extraction from complete documents, eliminating retrieval inconsistencies while optimizing for in-depth single-document analysis. While this delivers higher accuracy for individual documents, ContextGem does not currently support cross-document querying or corpus-wide retrieval - for these use cases, modern RAG systems (e.g., LlamaIndex, Haystack) remain more appropriate.

Read more on [how ContextGem works](https://contextgem.dev/how_it_works.html) in the documentation.


## 🤖 Supported LLMs

ContextGem supports both cloud-based and local LLMs through [LiteLLM](https://github.com/BerriAI/litellm) integration:
- **Cloud LLMs**: OpenAI, Anthropic, Google, Azure OpenAI, and more
- **Local LLMs**: Run models locally using providers like Ollama, LM Studio, etc.
- **Model Architectures**: Works with both reasoning/CoT-capable (e.g. o4-mini) and non-reasoning models (e.g. gpt-4.1)
- **Simple API**: Unified interface for all LLMs with easy provider switching


## ⚡ Optimizations

ContextGem documentation offers guidance on optimization strategies to maximize performance, minimize costs, and enhance extraction accuracy:

- [Optimizing for Accuracy](https://contextgem.dev/optimizations/optimization_accuracy.html)
- [Optimizing for Speed](https://contextgem.dev/optimizations/optimization_speed.html)
- [Optimizing for Cost](https://contextgem.dev/optimizations/optimization_cost.html)
- [Dealing with Long Documents](https://contextgem.dev/optimizations/optimization_long_docs.html)
- [Choosing the Right LLM(s)](https://contextgem.dev/optimizations/optimization_choosing_llm.html)


## 💾 Serializing results

ContextGem allows you to save and load Document objects, pipelines, and LLM configurations with built-in serialization methods:

- Save processed documents to avoid repeating expensive LLM calls
- Transfer extraction results between systems
- Persist pipeline and LLM configurations for later reuse

Learn more about [serialization options](https://contextgem.dev/serialization.html) in the documentation.


## 📚 Documentation

Full documentation is available at [contextgem.dev](https://contextgem.dev).

A raw text version of the full documentation is available at [`docs/docs-raw-for-llm.txt`](https://github.com/shcherbak-ai/contextgem/blob/main/docs/docs-raw-for-llm.txt). This file is automatically generated and contains all documentation in a format optimized for LLM ingestion (e.g. for Q&A).


## 💬 Community

If you have a feature request or a bug report, feel free to [open an issue](https://github.com/shcherbak-ai/contextgem/issues/new) on GitHub. If you'd like to discuss a topic or get general advice on using ContextGem for your project, start a thread in [GitHub Discussions](https://github.com/shcherbak-ai/contextgem/discussions/new/).


## 🤝 Contributing

We welcome contributions from the community - whether it's fixing a typo or developing a completely new feature! To get started, please check out our [Contributor Guidelines](https://github.com/shcherbak-ai/contextgem/blob/main/CONTRIBUTING.md).


## 🔐 Security

This project is automatically scanned for security vulnerabilities using [CodeQL](https://codeql.github.com/). We also use [Snyk](https://snyk.io) as needed for supplementary dependency checks.

See [SECURITY](https://github.com/shcherbak-ai/contextgem/blob/main/SECURITY.md) file for details.


## 💖 Acknowledgements

ContextGem relies on these excellent open-source packages:

- [pydantic](https://github.com/pydantic/pydantic): The gold standard for data validation
- [Jinja2](https://github.com/pallets/jinja): Fast, expressive template engine that powers our dynamic prompt rendering
- [litellm](https://github.com/BerriAI/litellm): Unified interface to multiple LLM providers with seamless provider switching
- [wtpsplit](https://github.com/segment-any-text/wtpsplit): State-of-the-art text segmentation tool
- [loguru](https://github.com/Delgan/loguru): Simple yet powerful logging that enhances debugging and observability
- [python-ulid](https://github.com/mdomke/python-ulid): Efficient ULID generation
- [PyTorch](https://github.com/pytorch/pytorch): Industry-standard machine learning framework
- [aiolimiter](https://github.com/mjpieters/aiolimiter): Powerful rate limiting for async operations


## 🌱 Support the project

ContextGem is just getting started, and your support means the world to us! If you find ContextGem useful, the best way to help is by sharing it with others and giving the project a ⭐. Your feedback and contributions are what make this project grow!


## 📄 License & Contact

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/shcherbak-ai/contextgem/blob/main/LICENSE) and [NOTICE](https://github.com/shcherbak-ai/contextgem/blob/main/NOTICE) files for details.

Copyright © 2025 [Shcherbak AI AS](https://shcherbak.ai), an AI engineering company building tools for AI/ML/NLP developers.

Shcherbak AI is now part of Microsoft for Startups.

[Connect with us on LinkedIn](https://www.linkedin.com/in/sergii-shcherbak-10068866/) for questions or collaboration ideas.

Built with ❤️ in Oslo, Norway.
