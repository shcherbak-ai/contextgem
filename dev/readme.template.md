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
[![DeepWiki](https://img.shields.io/static/v1?label=DeepWiki&message=Chat%20with%20Code&labelColor=%23283593&color=%237E57C2&style=flat-square)](https://deepwiki.com/shcherbak-ai/contextgem)
[![GitHub latest commit](https://img.shields.io/github/last-commit/shcherbak-ai/contextgem?label=latest%20commit)](https://github.com/shcherbak-ai/contextgem/commits/main)

<img src="https://contextgem.dev/_static/tab_solid.png" alt="ContextGem: 2nd Product of the week" width="250">
<br/><br/>

ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code.

---


## 💎 Why ContextGem?

Most popular LLM frameworks for extracting structured data from documents require extensive boilerplate code to extract even basic information. This significantly increases development time and complexity.

ContextGem addresses this challenge by providing a flexible, intuitive framework that extracts structured data and insights from documents with minimal effort. Complex, most time-consuming parts are handled with **powerful abstractions**, eliminating boilerplate code and reducing development overhead.

📖 Read more on the project [motivation](https://contextgem.dev/motivation.html) in the documentation.


## ⭐ Key features

{{FEATURE_TABLE}}

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

> **⚡ v0.5.0+**: ContextGem now installs 7.5x faster with minimal dependencies (no torch/transformers required), making it easier to integrate into existing ML environments.


## 🚀 Quick start

```python
{{QUICKSTART_CONCEPT}}
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shcherbak-ai/contextgem/blob/main/dev/notebooks/readme/quickstart_concept.ipynb)

---

### 📚 More Examples

**Basic usage:**
- [Aspect Extraction from Document](https://contextgem.dev/quickstart.html#aspect-extraction-from-document)
- [Extracting Aspect with Sub-Aspects](https://contextgem.dev/quickstart.html#extracting-aspect-with-sub-aspects)
- [Concept Extraction from Aspect](https://contextgem.dev/quickstart.html#concept-extraction-from-aspect)
- [Concept Extraction from Document (text)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-text)
- [Concept Extraction from Document (vision)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-vision)
- [LLM chat interface](https://contextgem.dev/quickstart.html#lightweight-llm-chat-interface)

**Advanced usage:**
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
{{DOCX_CONVERTER}}
```

📖 Learn more about [DOCX converter features](https://contextgem.dev/converters/docx.html) in the documentation.

## 🎯 Focused document analysis

ContextGem leverages LLMs' long context windows to deliver superior extraction accuracy from individual documents. Unlike RAG approaches that often [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f), ContextGem capitalizes on continuously expanding context capacity, evolving LLM capabilities, and decreasing costs. This focused approach enables direct information extraction from complete documents, eliminating retrieval inconsistencies while optimizing for in-depth single-document analysis. While this delivers higher accuracy for individual documents, ContextGem does not currently support cross-document querying or corpus-wide retrieval - for these use cases, modern RAG systems (e.g., LlamaIndex, Haystack) remain more appropriate.

📖 Read more on [how ContextGem works](https://contextgem.dev/how_it_works.html) in the documentation.

## 🤖 Supported LLMs

ContextGem supports both cloud-based and local LLMs through [LiteLLM](https://github.com/BerriAI/litellm) integration:
- **Cloud LLMs**: OpenAI, Anthropic, Google, Azure OpenAI, and more
- **Local LLMs**: Run models locally using providers like Ollama, LM Studio, etc.
- **Model Architectures**: Works with both reasoning/CoT-capable (e.g. o4-mini) and non-reasoning models (e.g. gpt-4.1)
- **Simple API**: Unified interface for all LLMs with easy provider switching

📖 Learn more about [supported LLM providers and models](https://contextgem.dev/llms/supported_llms.html), how to [configure LLMs](https://contextgem.dev/llms/llm_config.html), and [LLM extraction methods](https://contextgem.dev/llms/llm_extraction_methods.html) in the documentation.

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

This project is automatically scanned for security vulnerabilities using [CodeQL](https://codeql.github.com/). We also use [Snyk](https://snyk.io) as needed for supplementary dependency checks.

🛡️ **Security policy:** See [SECURITY](https://github.com/shcherbak-ai/contextgem/blob/main/SECURITY.md) file for details.

## 💖 Acknowledgements

ContextGem relies on these excellent open-source packages:

- [pydantic](https://github.com/pydantic/pydantic): The gold standard for data validation
- [Jinja2](https://github.com/pallets/jinja): Fast, expressive template engine that powers our dynamic prompt rendering
- [litellm](https://github.com/BerriAI/litellm): Unified interface to multiple LLM providers with seamless provider switching
- [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite): Lightweight version of [wtpsplit](https://github.com/segment-any-text/wtpsplit) for state-of-the-art text segmentation using wtpsplit's SaT models
- [loguru](https://github.com/Delgan/loguru): Simple yet powerful logging that enhances debugging and observability
- [python-ulid](https://github.com/mdomke/python-ulid): Efficient ULID generation
- [aiolimiter](https://github.com/mjpieters/aiolimiter): Powerful rate limiting for async operations


## 🌱 Support the project

ContextGem is just getting started, and your support means the world to us! 

⭐ **Star the project** if you find ContextGem useful  
📢 **Share it** with others who might benefit  
🔧 **Contribute** with feedback, issues, or code improvements

Your engagement is what makes this project grow!

## 📄 License & Contact

**License:** Apache 2.0 License - see the [LICENSE](https://github.com/shcherbak-ai/contextgem/blob/main/LICENSE) and [NOTICE](https://github.com/shcherbak-ai/contextgem/blob/main/NOTICE) files for details.

**Copyright:** © 2025 [Shcherbak AI AS](https://shcherbak.ai), an AI engineering company building tools for AI/ML/NLP developers.

**Connect:** [LinkedIn](https://www.linkedin.com/in/sergii-shcherbak-10068866/) for questions or collaboration ideas.

Built with ❤️ in Oslo, Norway.
