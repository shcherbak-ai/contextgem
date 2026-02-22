![ContextGem](https://contextgem.dev/_static/contextgem_readme_header.png "ContextGem - Effortless LLM extraction from documents")

# ContextGem: Effortless LLM extraction from documents

|          |        |
|----------|--------|
| **Package** | [![PyPI](https://img.shields.io/pypi/v/contextgem?logo=pypi&label=PyPi&logoColor=gold&style=flat-square)](https://pypi.org/project/contextgem/) [![PyPI Downloads](https://static.pepy.tech/badge/contextgem/month)](https://pepy.tech/projects/contextgem) [![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=gold&style=flat-square)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0) |
| **Quality** | [![tests](https://img.shields.io/github/actions/workflow/status/shcherbak-ai/contextgem/ci-tests.yml?branch=main&style=flat-square&label=tests)](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml) [![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SergiiShcherbak/daaee00e1dfff7a29ca10a922ec3becd/raw/coverage.json&style=flat-square)](https://github.com/shcherbak-ai/contextgem/actions) [![CodeQL](https://img.shields.io/github/actions/workflow/status/shcherbak-ai/contextgem/codeql.yml?branch=main&style=flat-square&label=CodeQL)](https://github.com/shcherbak-ai/contextgem/actions/workflows/codeql.yml) [![bandit security](https://img.shields.io/github/actions/workflow/status/shcherbak-ai/contextgem/bandit-security.yml?branch=main&style=flat-square&label=bandit)](https://github.com/shcherbak-ai/contextgem/actions/workflows/bandit-security.yml) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10489/badge?1)](https://www.bestpractices.dev/projects/10489) |
| **Tools** | [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff) [![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json&style=flat-square)](https://pydantic.dev) [![pyright](https://img.shields.io/badge/pyright-checked-blue?style=flat-square)](https://github.com/microsoft/pyright) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue?logo=pre-commit&logoColor=white&style=flat-square)](https://github.com/pre-commit/pre-commit) [![deptry](https://img.shields.io/badge/deptry-checked-blue?style=flat-square)](https://github.com/fpgmaas/deptry) [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg?style=flat-square)](https://github.com/pypa/hatch) |
| **Docs** | [![docs](https://img.shields.io/github/actions/workflow/status/shcherbak-ai/contextgem/docs.yml?branch=main&style=flat-square&label=docs)](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml) [![documentation](https://img.shields.io/badge/docs-latest-blue?style=flat-square)](https://shcherbak-ai.github.io/contextgem/) ![Docstring Coverage](https://contextgem.dev/_static/interrogate-badge.svg) [![DeepWiki](https://img.shields.io/static/v1?label=DeepWiki&message=Chat%20with%20Code&labelColor=%23283593&color=%237E57C2&style=flat-square)](https://deepwiki.com/shcherbak-ai/contextgem) |
| **AI Agents** | [![AGENTS.md](https://img.shields.io/badge/AGENTS.md-✓-blue?style=flat-square)](https://agents.md) [![CLAUDE.md](https://img.shields.io/badge/CLAUDE.md-✓-blue?style=flat-square)](https://docs.anthropic.com/en/docs/claude-code/memory#claudemd) |
| **Community** | [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa?style=flat-square)](CODE_OF_CONDUCT.md) [![GitHub issues closed](https://img.shields.io/github/issues-closed/shcherbak-ai/contextgem?style=flat-square)](https://github.com/shcherbak-ai/contextgem/issues?q=is%3Aissue+is%3Aclosed) [![GitHub latest commit](https://img.shields.io/github/last-commit/shcherbak-ai/contextgem?label=latest%20commit&style=flat-square)](https://github.com/shcherbak-ai/contextgem/commits/main) |

<div align="center">
<img src="https://contextgem.dev/_static/tab_solid.png" alt="ContextGem: 2nd Product of the week" width="250">
</div>
<br/><br/>

ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code.

---

## 💎 Why ContextGem?

Most popular LLM frameworks for extracting structured data from documents require extensive boilerplate code to extract even basic information. This significantly increases development time and complexity.

ContextGem addresses this challenge by providing a flexible, intuitive framework that extracts structured data and insights from documents with minimal effort. The complex, most time-consuming parts are handled with **powerful abstractions**, eliminating boilerplate code and reducing development overhead.

📖 Read more on the project [motivation](https://contextgem.dev/motivation/) in the documentation.


## ⭐ Key features

{{FEATURE_TABLE}}

\* See [descriptions](https://contextgem.dev/motivation/#the-contextgem-solution) of ContextGem abstractions and [comparisons](https://contextgem.dev/vs_other_frameworks/) of specific implementation examples using ContextGem and other popular open-source LLM frameworks.

## 💡 What you can build

With **minimal code**, you can:

- **Extract structured data** from documents (text, images)
- **Identify and analyze key aspects** (topics, themes, categories) within documents ([learn more](https://contextgem.dev/aspects/aspects/))
- **Extract specific concepts** (entities, facts, conclusions, assessments) from documents ([learn more](https://contextgem.dev/concepts/supported_concepts/))
- **Build complex extraction workflows** through a simple, intuitive API
- **Create multi-level extraction pipelines** (aspects containing concepts, hierarchical aspects)

<br/>

![ContextGem extraction example](https://contextgem.dev/_static/readme_code_snippet.png "ContextGem extraction example")


## 📦 Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv add contextgem
```

Or using pip:

```bash
pip install -U contextgem
```


## 🚀 Quick start

The following example demonstrates how to use ContextGem to extract **anomalies** from a legal document - a complex concept that requires contextual understanding. Unlike traditional RAG approaches that might miss subtle inconsistencies, ContextGem analyzes the entire document context to identify content that doesn't belong, complete with source references and justifications.

```python
{{QUICKSTART_CONCEPT}}
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
<td>Create a Document that contains text and/or visual content representing your document (contract, invoice, report, CV, etc.), from which an LLM extracts information (aspects and/or concepts). <a href="https://contextgem.dev/documents/document_config/">Learn more</a></td>
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
<td>Define Aspects to extract text segments from the document (sections, topics, themes). You can organize content hierarchically and combine with concepts for comprehensive analysis. <a href="https://contextgem.dev/aspects/aspects/">Learn more</a></td>
<td>Define Concepts to extract specific data points with intelligent inference: entities, insights, structured objects, classifications, numerical calculations, dates, ratings, and assessments. <a href="https://contextgem.dev/concepts/supported_concepts/">Learn more</a></td>
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
<td>Create a reusable collection of predefined aspects and concepts that enables consistent extraction across multiple documents. <a href="https://contextgem.dev/pipelines/extraction_pipelines/">Learn more</a></td>
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
<td>Configure a cloud or local LLM that will extract aspects and/or concepts from the document. DocumentLLM supports fallback models and role-based task routing for optimal performance. <a href="https://contextgem.dev/llms/llm_extraction_methods/">Learn more</a></td>
<td>Configure a group of LLMs with unique roles for complex extraction workflows. You can route different aspects and/or concepts to specialized LLMs (e.g., simple extraction vs. reasoning tasks). <a href="https://contextgem.dev/llms/llm_config/#llm-groups">Learn more</a></td>
</tr>
</tbody>
</table>

```python
llm = DocumentLLM(
    model="openai/gpt-5-mini",  # or another provider/LLM
    api_key="...",
)
document = llm.extract_all(document)
# print(document.aspects[0].extracted_items)
# print(document.concepts[0].extracted_items)
```

📖 Learn more about ContextGem's [core components](https://contextgem.dev/how_it_works/) and their practical examples in the documentation.

## 📚 Usage Examples

🌟 **Basic usage:**
- [Aspect Extraction from Document](https://contextgem.dev/quickstart/#aspect-extraction-from-document)
- [Extracting Aspect with Sub-Aspects](https://contextgem.dev/quickstart/#extracting-aspect-with-sub-aspects)
- [Concept Extraction from Aspect](https://contextgem.dev/quickstart/#concept-extraction-from-aspect)
- [Concept Extraction from Document (text)](https://contextgem.dev/quickstart/#concept-extraction-from-document-text)
- [Concept Extraction from Document (vision)](https://contextgem.dev/quickstart/#concept-extraction-from-document-vision)
- [LLM chat interface with tool calling](https://contextgem.dev/quickstart/#lightweight-llm-chat-interface)

🚀 **Advanced usage:**
- [Extracting Aspects Containing Concepts](https://contextgem.dev/advanced_usage/#extracting-aspects-with-concepts)
- [Extracting Aspects and Concepts from a Document](https://contextgem.dev/advanced_usage/#extracting-aspects-and-concepts-from-a-document)
- [Using a Multi-LLM Pipeline to Extract Data from Several Documents](https://contextgem.dev/advanced_usage/#using-a-multi-llm-pipeline-to-extract-data-from-several-documents)


## 🔄 Document converters

To create a ContextGem document for LLM analysis, you can either pass raw text directly, or use built-in converters that handle various file formats.

### 📄 DOCX converter

 ContextGem provides a built-in converter to easily transform DOCX files into LLM-ready data.

- **Comprehensive extraction of document elements**: paragraphs, headings, lists, tables, comments, footnotes, textboxes, headers/footers, links, embedded images, and inline formatting
- **Document structure preservation** with rich metadata for improved LLM analysis
- **Built-in converter** that directly processes Word XML

> 🚀 **Performance improvement in v0.17.1**: DOCX converter now converts files **~2X faster**.

```python
{{DOCX_CONVERTER}}
```

📖 Learn more about [DOCX converter features](https://contextgem.dev/converters/docx/) in the documentation.


## 🎯 Focused document analysis

ContextGem leverages LLMs' long context windows to deliver superior extraction accuracy from individual documents. Unlike RAG approaches that often [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f), ContextGem capitalizes on continuously expanding context capacity, evolving LLM capabilities, and decreasing costs. This focused approach enables direct information extraction from complete documents, eliminating retrieval inconsistencies while optimizing for in-depth single-document analysis. While this delivers higher accuracy for individual documents, ContextGem does not currently support cross-document querying or corpus-wide retrieval - for these use cases, modern RAG frameworks (e.g., LlamaIndex, Haystack) remain more appropriate.

📖 Read more on [how ContextGem works](https://contextgem.dev/how_it_works/) in the documentation.

## 🤖 Supported LLMs

ContextGem supports both cloud-based and local LLMs through [LiteLLM](https://github.com/BerriAI/litellm) integration:
- **Cloud LLMs**: OpenAI, Anthropic, Google, Azure OpenAI, xAI, and more
- **Local LLMs**: Run models locally using providers like Ollama, LM Studio, etc.
- **Model Architectures**: Works with both reasoning/CoT-capable (e.g. gpt-5) and non-reasoning models (e.g. gpt-4.1)
- **Simple API**: Unified interface for all LLMs with easy provider switching

> **💡 Model Selection Note:** For reliable structured extraction, we recommend using models with performance equivalent to or exceeding `gpt-4o-mini`. Smaller models (such as 8B parameter models) may struggle with ContextGem's detailed extraction instructions. If you encounter issues with smaller models, see our [troubleshooting guide](https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting/) for potential solutions.

📖 Learn more about [supported LLM providers and models](https://contextgem.dev/llms/supported_llms/), how to [configure LLMs](https://contextgem.dev/llms/llm_config/), and [LLM extraction methods](https://contextgem.dev/llms/llm_extraction_methods/) in the documentation.

## ⚡ Optimizations

ContextGem documentation offers guidance on optimization strategies to maximize performance, minimize costs, and enhance extraction accuracy:

- [Optimizing for Accuracy](https://contextgem.dev/optimizations/optimization_accuracy/)
- [Optimizing for Speed](https://contextgem.dev/optimizations/optimization_speed/)
- [Optimizing for Cost](https://contextgem.dev/optimizations/optimization_cost/)
- [Dealing with Long Documents](https://contextgem.dev/optimizations/optimization_long_docs/)
- [Choosing the Right LLM(s)](https://contextgem.dev/optimizations/optimization_choosing_llm/)
- [Troubleshooting Issues with Small Models](https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting/)


## 💾 Serializing results

ContextGem allows you to save and load Document objects, pipelines, and LLM configurations with built-in serialization methods:

- Save processed documents to avoid repeating expensive LLM calls
- Transfer extraction results between systems
- Persist pipeline and LLM configurations for later reuse

📖 Learn more about [serialization options](https://contextgem.dev/serialization/) in the documentation.


## 📚 Documentation

📖 **Full documentation:** [contextgem.dev](https://contextgem.dev)

> **⚠️ Official Documentation Notice:** [https://contextgem.dev/](https://contextgem.dev/) is the only official source of ContextGem documentation. Please be aware of unauthorized copies or mirrors that may contain outdated or incorrect information.

📄 **Raw documentation for LLMs:** Available at [`docs/source/llms.txt`](https://github.com/shcherbak-ai/contextgem/blob/main/docs/source/llms.txt) - automatically generated, optimized for LLM ingestion.

🤖 **AI-powered code exploration:** [DeepWiki](https://deepwiki.com/shcherbak-ai/contextgem) provides visual architecture maps and natural language Q&A for the codebase.

📈 **Change history:** See the [CHANGELOG](https://github.com/shcherbak-ai/contextgem/blob/main/CHANGELOG.md) for version history, improvements, and bug fixes.

## 💬 Community

🐛 **Found a bug or have a feature request?** [Open an issue](https://github.com/shcherbak-ai/contextgem/issues/new) on GitHub.

💭 **Need help or want to discuss?** Start a thread in [GitHub Discussions](https://github.com/shcherbak-ai/contextgem/discussions/new/).

## 🤝 Contributing

We welcome contributions from the community - whether it's fixing a typo or developing a completely new feature!

📋 **Get started:** Check out our [Contributor Guidelines](https://github.com/shcherbak-ai/contextgem/blob/main/CONTRIBUTING.md).

## 🤖 AI Agent-Friendly

This repository follows modern development practices with first-class support for AI coding assistants:

- **[AGENTS.md](https://github.com/shcherbak-ai/contextgem/blob/main/AGENTS.md)** - Guidelines for AI assistants working with this codebase ([agents.md standard](https://agents.md))
- **[CLAUDE.md](https://github.com/shcherbak-ai/contextgem/blob/main/CLAUDE.md)** - Configuration for [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)

These files provide AI assistants with project-specific context, coding conventions, architecture patterns, and contribution workflows - enabling more accurate and consistent AI-assisted development.

## 🔐 Security

This project is automatically scanned for security vulnerabilities using multiple security tools:

- **[CodeQL](https://codeql.github.com/)** - GitHub's semantic code analysis engine for vulnerability detection
- **[Bandit](https://github.com/PyCQA/bandit)** - Python security linter for common security issues  
- **[Snyk](https://snyk.io)** - Dependency vulnerability monitoring (used as needed)

🛡️ **Security policy:** See [SECURITY](https://github.com/shcherbak-ai/contextgem/blob/main/SECURITY.md) file for details.

## 💖 Acknowledgements

ContextGem relies on these excellent open-source packages:

- [aiolimiter](https://github.com/mjpieters/aiolimiter): Powerful rate limiting for async operations
- [colorlog](https://github.com/borntyping/python-colorlog): Colored formatter for Python's logging module
- [docstring-parser](https://github.com/rr-/docstring_parser): Docstring parsing for auto-generating tool schemas
- [fastjsonschema](https://github.com/horejsek/python-fastjsonschema): Ultra-fast JSON schema validation
- [genai-prices](https://github.com/pydantic/genai-prices): LLM pricing data and utilities (by Pydantic) to automatically estimate costs
- [Jinja2](https://github.com/pallets/jinja): Fast, expressive, extensible templating engine used for prompt rendering
- [litellm](https://github.com/BerriAI/litellm): Unified interface to multiple LLM providers with seamless provider switching
- [lxml](https://github.com/lxml/lxml): High-performance XML processing library for parsing DOCX document structure
- [pillow](https://github.com/python-pillow/Pillow): Image processing library for local model image handling
- [pydantic](https://github.com/pydantic/pydantic): The gold standard for data validation
- [python-ulid](https://github.com/mdomke/python-ulid): Efficient ULID generation for unique object identification
- [tenacity](https://github.com/jd/tenacity): General-purpose retry library for Python
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
