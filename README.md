![ContextGem](https://contextgem.dev/_static/contextgem_readme_header.png "ContextGem - Effortless LLM extraction from documents")

# ContextGem: Effortless LLM extraction from documents

|          |        |
|----------|--------|
| **Package** | [![PyPI](https://img.shields.io/pypi/v/contextgem?logo=pypi&label=PyPi&logoColor=gold)](https://pypi.org/project/contextgem/) [![PyPI Downloads](https://static.pepy.tech/badge/contextgem/month)](https://pepy.tech/projects/contextgem) [![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=gold)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/License-Apache_2.0-bright.svg)](https://opensource.org/licenses/Apache-2.0) |
| **Quality** | [![tests](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml) [![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SergiiShcherbak/daaee00e1dfff7a29ca10a922ec3becd/raw/coverage.json)](https://github.com/shcherbak-ai/contextgem/actions) [![CodeQL](https://github.com/shcherbak-ai/contextgem/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/codeql.yml) [![bandit security](https://github.com/shcherbak-ai/contextgem/actions/workflows/bandit-security.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/bandit-security.yml) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10489/badge?1)](https://www.bestpractices.dev/projects/10489) |
| **Tools** | [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev) [![pyright](https://img.shields.io/badge/pyright-checked-blue)](https://github.com/microsoft/pyright) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![deptry](https://img.shields.io/badge/deptry-checked-blue)](https://github.com/fpgmaas/deptry) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) |
| **Docs** | [![docs](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml) [![documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://shcherbak-ai.github.io/contextgem/) ![Docstring Coverage](https://contextgem.dev/_static/interrogate-badge.svg) [![DeepWiki](https://img.shields.io/static/v1?label=DeepWiki&message=Chat%20with%20Code&labelColor=%23283593&color=%237E57C2&style=flat-square)](https://deepwiki.com/shcherbak-ai/contextgem) |
| **Community** | [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md) [![GitHub issues closed](https://img.shields.io/github/issues-closed/shcherbak-ai/contextgem)](https://github.com/shcherbak-ai/contextgem/issues?q=is%3Aissue+is%3Aclosed) [![GitHub latest commit](https://img.shields.io/github/last-commit/shcherbak-ai/contextgem?label=latest%20commit)](https://github.com/shcherbak-ai/contextgem/commits/main) |

<div align="center">
<img src="https://contextgem.dev/_static/tab_solid.png" alt="ContextGem: 2nd Product of the week" width="250">
</div>
<br/><br/>

ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code.

---

## 💎 Why ContextGem?

Most popular LLM frameworks for extracting structured data from documents require extensive boilerplate code to extract even basic information. This significantly increases development time and complexity.

ContextGem addresses this challenge by providing a flexible, intuitive framework that extracts structured data and insights from documents with minimal effort. The complex, most time-consuming parts are handled with **powerful abstractions**, eliminating boilerplate code and reducing development overhead.

📖 Read more on the project [motivation](https://contextgem.dev/motivation.html) in the documentation.


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
                Neural segmentation (using wtpsplit's SaT models)
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

## 💡 What you can build

With **minimal code**, you can:

- **Extract structured data** from documents (text, images)
- **Identify and analyze key aspects** (topics, themes, categories) within documents ([learn more](https://contextgem.dev/aspects/aspects.html))
- **Extract specific concepts** (entities, facts, conclusions, assessments) from documents ([learn more](https://contextgem.dev/concepts/supported_concepts.html))
- **Build complex extraction workflows** through a simple, intuitive API
- **Create multi-level extraction pipelines** (aspects containing concepts, hierarchical aspects)

<br/>

![ContextGem extraction example](https://contextgem.dev/_static/readme_code_snippet.png "ContextGem extraction example")


## 📦 Installation

```bash
pip install -U contextgem
```


## 🚀 Quick start

The following example demonstrates how to use ContextGem to extract **anomalies** from a legal document - a complex concept that requires contextual understanding. Unlike traditional RAG approaches that might miss subtle inconsistencies, ContextGem analyzes the entire document context to identify content that doesn't belong, complete with source references and justifications.

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
        "Time-traveling dinosaurs will review all deliverables before acceptance.\n"  # 💎 another anomaly
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
anomalies_concept = doc.concepts[0]
# or `doc.get_concept_by_name("Anomalies")`
for item in anomalies_concept.extracted_items:
    print("Anomaly:")
    print(f"  {item.value}")
    print("Justification:")
    print(f"  {item.justification}")
    print("Reference paragraphs:")
    for p in item.reference_paragraphs:
        print(f"  - {p.raw_text}")
    print("Reference sentences:")
    for s in item.reference_sentences:
        print(f"  - {s.raw_text}")
    print()

```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/readme/quickstart_concept.ipynb)

---


## 🧠 How it works

### 📝 Step 1: Define extraction context

<table>
<thead>
<tr>
<th width="100%" align="left">📄 <strong>Document</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Create a Document that contains text and/or visual content representing your document (contract, invoice, report, CV, etc.), from which an LLM extracts information (aspects and/or concepts). <a href="https://contextgem.dev/documents/document_config.html">Learn more</a></td>
</tr>
</tbody>
</table>

```python
document = Document(raw_text="Non-Disclosure Agreement...")
```

### 🎯 Step 2: Define what to extract

<table>
<thead>
<tr>
<th width="50%" align="left">🔍 <strong>Aspects</strong></th>
<th width="50%" align="left">💡 <strong>Concepts</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Define Aspects to extract text segments from the document (sections, topics, themes). You can organize content hierarchically and combine with concepts for comprehensive analysis. <a href="https://contextgem.dev/aspects/aspects.html">Learn more</a></td>
<td>Define Concepts to extract specific data points with intelligent inference: entities, insights, structured objects, classifications, numerical calculations, dates, ratings, and assessments. <a href="https://contextgem.dev/concepts/supported_concepts.html">Learn more</a></td>
</tr>
</tbody>
</table>

```python
# Extract document sections
aspect = Aspect(
    name="Term and termination",
    description="Clauses on contract term and termination",
)
# Extract specific data points
concept = BooleanConcept(
    name="NDA check",
    description="Is the contract an NDA?",
)
# Add these to the document instance for further extraction
document.add_aspects([aspect])
document.add_concepts([concept])
```

<table>
<thead>
<tr>
<th width="100%" align="left">🔄 <i>Alternative</i>: Configure  <strong>Extraction Pipeline</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Create a reusable collection of predefined aspects and concepts that enables consistent extraction across multiple documents. <a href="https://contextgem.dev/pipelines/extraction_pipelines.html">Learn more</a></td>
</tr>
</tbody>
</table>

### 🧠 Step 3: Run LLM extraction

<table>
<thead>
<tr>
<th width="50%" align="left">🤖 <strong>LLM</strong></th>
<th width="50%" align="left">🤖🤖 <i>Alternative</i>: <strong>LLM Group (advanced)</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>Configure a cloud or local LLM that will extract aspects and/or concepts from the document. DocumentLLM supports fallback models and role-based task routing for optimal performance. <a href="https://contextgem.dev/llms/llm_extraction_methods.html">Learn more</a></td>
<td>Configure a group of LLMs with unique roles for complex extraction workflows. You can route different aspects and/or concepts to specialized LLMs (e.g., simple extraction vs. reasoning tasks). <a href="https://contextgem.dev/llms/llm_config.html#llm-groups">Learn more</a></td>
</tr>
</tbody>
</table>

```python
llm = DocumentLLM(
    model="openai/gpt-4.1-mini",  # or another provider/LLM
    api_key="...",
)
document = llm.extract_all(document)
# print(document.aspects[0].extracted_items)
# print(document.concepts[0].extracted_items)
```

📖 Learn more about ContextGem's [core components](https://contextgem.dev/how_it_works.html) and their practical examples in the documentation.

## 📚 Usage Examples

🌟 **Basic usage:**
- [Aspect Extraction from Document](https://contextgem.dev/quickstart.html#aspect-extraction-from-document)
- [Extracting Aspect with Sub-Aspects](https://contextgem.dev/quickstart.html#extracting-aspect-with-sub-aspects)
- [Concept Extraction from Aspect](https://contextgem.dev/quickstart.html#concept-extraction-from-aspect)
- [Concept Extraction from Document (text)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-text)
- [Concept Extraction from Document (vision)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-vision)
- [LLM chat interface](https://contextgem.dev/quickstart.html#lightweight-llm-chat-interface)

🚀 **Advanced usage:**
- [Extracting Aspects Containing Concepts](https://contextgem.dev/advanced_usage.html#extracting-aspects-with-concepts)
- [Extracting Aspects and Concepts from a Document](https://contextgem.dev/advanced_usage.html#extracting-aspects-and-concepts-from-a-document)
- [Using a Multi-LLM Pipeline to Extract Data from Several Documents](https://contextgem.dev/advanced_usage.html#using-a-multi-llm-pipeline-to-extract-data-from-several-documents)


## 🔄 Document converters

To create a ContextGem document for LLM analysis, you can either pass raw text directly, or use built-in converters that handle various file formats.

### 📄 DOCX converter

 ContextGem provides a built-in converter to easily transform DOCX files into LLM-ready data.

- **Comprehensive extraction of document elements**: paragraphs, headings, lists, tables, comments, footnotes, textboxes, headers/footers, links, embedded images, and inline formatting
- **Document structure preservation** with rich metadata for improved LLM analysis
- **Built-in converter** that directly processes Word XML

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

# Perform data extraction on the resulting Document object
# document.add_aspects(...)
# document.add_concepts(...)
# llm.extract_all(document)

# You can also use DocxConverter instance as a standalone text extractor
docx_text = converter.convert_to_text_format(
    "path/to/document.docx",
    output_format="markdown",  # or "raw"
)

```

📖 Learn more about [DOCX converter features](https://contextgem.dev/converters/docx.html) in the documentation.


## 🎯 Focused document analysis

ContextGem leverages LLMs' long context windows to deliver superior extraction accuracy from individual documents. Unlike RAG approaches that often [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f), ContextGem capitalizes on continuously expanding context capacity, evolving LLM capabilities, and decreasing costs. This focused approach enables direct information extraction from complete documents, eliminating retrieval inconsistencies while optimizing for in-depth single-document analysis. While this delivers higher accuracy for individual documents, ContextGem does not currently support cross-document querying or corpus-wide retrieval - for these use cases, modern RAG systems (e.g., LlamaIndex, Haystack) remain more appropriate.

📖 Read more on [how ContextGem works](https://contextgem.dev/how_it_works.html) in the documentation.

## 🤖 Supported LLMs

ContextGem supports both cloud-based and local LLMs through [LiteLLM](https://github.com/BerriAI/litellm) integration:
- **Cloud LLMs**: OpenAI, Anthropic, Google, Azure OpenAI, xAI, and more
- **Local LLMs**: Run models locally using providers like Ollama, LM Studio, etc.
- **Model Architectures**: Works with both reasoning/CoT-capable (e.g. o1-mini) and non-reasoning models (e.g. gpt-4o)
- **Simple API**: Unified interface for all LLMs with easy provider switching

> **💡 Model Selection Note:** For reliable structured extraction, we recommend using models with performance equivalent to or exceeding `gpt-4o-mini`. Smaller models (such as 8B parameter models) may struggle with ContextGem's detailed extraction instructions. If you encounter issues with smaller models, see our [troubleshooting guide](https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting.html) for potential solutions.

📖 Learn more about [supported LLM providers and models](https://contextgem.dev/llms/supported_llms.html), how to [configure LLMs](https://contextgem.dev/llms/llm_config.html), and [LLM extraction methods](https://contextgem.dev/llms/llm_extraction_methods.html) in the documentation.

## ⚡ Optimizations

ContextGem documentation offers guidance on optimization strategies to maximize performance, minimize costs, and enhance extraction accuracy:

- [Optimizing for Accuracy](https://contextgem.dev/optimizations/optimization_accuracy.html)
- [Optimizing for Speed](https://contextgem.dev/optimizations/optimization_speed.html)
- [Optimizing for Cost](https://contextgem.dev/optimizations/optimization_cost.html)
- [Dealing with Long Documents](https://contextgem.dev/optimizations/optimization_long_docs.html)
- [Choosing the Right LLM(s)](https://contextgem.dev/optimizations/optimization_choosing_llm.html)
- [Troubleshooting Issues with Small Models](https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting.html)


## 💾 Serializing results

ContextGem allows you to save and load Document objects, pipelines, and LLM configurations with built-in serialization methods:

- Save processed documents to avoid repeating expensive LLM calls
- Transfer extraction results between systems
- Persist pipeline and LLM configurations for later reuse

📖 Learn more about [serialization options](https://contextgem.dev/serialization.html) in the documentation.


## 📚 Documentation

📖 **Full documentation:** [contextgem.dev](https://contextgem.dev)

📄 **Raw documentation for LLMs:** Available at [`docs/docs-raw-for-llm.txt`](https://github.com/shcherbak-ai/contextgem/blob/main/docs/docs-raw-for-llm.txt) - automatically generated, optimized for LLM ingestion.

🤖 **AI-powered code exploration:** [DeepWiki](https://deepwiki.com/shcherbak-ai/contextgem) provides visual architecture maps and natural language Q&A for the codebase.

📈 **Change history:** See the [CHANGELOG](https://github.com/shcherbak-ai/contextgem/blob/main/CHANGELOG.md) for version history, improvements, and bug fixes.

## 💬 Community

🐛 **Found a bug or have a feature request?** [Open an issue](https://github.com/shcherbak-ai/contextgem/issues/new) on GitHub.

💭 **Need help or want to discuss?** Start a thread in [GitHub Discussions](https://github.com/shcherbak-ai/contextgem/discussions/new/).

## 🤝 Contributing

We welcome contributions from the community - whether it's fixing a typo or developing a completely new feature! 

📋 **Get started:** Check out our [Contributor Guidelines](https://github.com/shcherbak-ai/contextgem/blob/main/CONTRIBUTING.md).

## 🔐 Security

This project is automatically scanned for security vulnerabilities using multiple security tools:

- **[CodeQL](https://codeql.github.com/)** - GitHub's semantic code analysis engine for vulnerability detection
- **[Bandit](https://github.com/PyCQA/bandit)** - Python security linter for common security issues  
- **[Snyk](https://snyk.io)** - Dependency vulnerability monitoring (used as needed)

🛡️ **Security policy:** See [SECURITY](https://github.com/shcherbak-ai/contextgem/blob/main/SECURITY.md) file for details.

## 💖 Acknowledgements

ContextGem relies on these excellent open-source packages:

- [aiolimiter](https://github.com/mjpieters/aiolimiter): Powerful rate limiting for async operations
- [genai-prices](https://github.com/pydantic/genai-prices): LLM pricing data and utilities (by Pydantic) to automatically estimate costs
- [Jinja2](https://github.com/pallets/jinja): Fast, expressive, extensible templating engine used for prompt rendering
- [litellm](https://github.com/BerriAI/litellm): Unified interface to multiple LLM providers with seamless provider switching
- [loguru](https://github.com/Delgan/loguru): Simple yet powerful logging that enhances debugging and observability
- [lxml](https://github.com/lxml/lxml): High-performance XML processing library for parsing DOCX document structure
- [pillow](https://github.com/python-pillow/Pillow): Image processing library for local model image handling
- [pydantic](https://github.com/pydantic/pydantic): The gold standard for data validation
- [python-ulid](https://github.com/mdomke/python-ulid): Efficient ULID generation for unique object identification
- [typing-extensions](https://github.com/python/typing_extensions): Backports of the latest typing features for enhanced type annotations
- [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite): Lightweight version of [wtpsplit](https://github.com/segment-any-text/wtpsplit) for state-of-the-art paragraph/sentence segmentation using wtpsplit's SaT models


## 🌱 Support the project

ContextGem is just getting started, and your support means the world to us! 

⭐ **Star the project** if you find ContextGem useful  
📢 **Share it** with others who might benefit  
🔧 **Contribute** with feedback, issues, or code improvements

Your engagement is what makes this project grow!


## 📄 License & Contact

**License:** Apache 2.0 License - see the [LICENSE](https://github.com/shcherbak-ai/contextgem/blob/main/LICENSE) and [NOTICE](https://github.com/shcherbak-ai/contextgem/blob/main/NOTICE) files for details.

**Copyright:** © 2025 [Shcherbak AI AS](https://shcherbak.ai), an AI engineering company building tools for AI/ML/NLP developers.

**Connect:** [LinkedIn](https://www.linkedin.com/in/sergii-shcherbak-10068866/) or [X](https://x.com/seshch) for questions or collaboration ideas.

Built with ❤️ in Oslo, Norway.
