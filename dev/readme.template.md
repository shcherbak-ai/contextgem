![ContextGem](https://contextgem.dev/_static/contextgem_poster.png "ContextGem - Easier and faster way to build LLM extraction workflows through powerful abstractions")

# ContextGem: Easier and faster way to build LLM extraction workflows

[![tests](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SergiiShcherbak/daaee00e1dfff7a29ca10a922ec3becd/raw/coverage.json)](https://github.com/shcherbak-ai/contextgem/actions)
[![docs](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/shcherbak-ai/contextgem/actions/workflows/docs.yml)
[![documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://shcherbak-ai.github.io/contextgem/)
[![License](https://img.shields.io/badge/License-Apache_2.0-bright.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-blue?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

ContextGem is an LLM framework for easier, faster extraction of structured data and insights from documents through powerful abstractions.


## 💎 Why ContextGem?

Most popular LLM frameworks for extracting structured data from documents require extensive boilerplate code to extract even basic information. This significantly increases development time and complexity.

ContextGem addresses this challenge by providing a flexible, intuitive framework that extracts structured data and insights from documents with minimal effort. Complex, most time-consuming parts are handled with **powerful abstractions**, eliminating boilerplate code and reducing development overhead.

Read more on the project [motivation](https://contextgem.dev/motivation.html) in the documentation.


## ⭐ Key features

{{FEATURE_TABLE}}

\* See [descriptions](https://contextgem.dev/motivation.html#the-contextgem-solution) of ContextGem abstractions and [comparisons](https://contextgem.dev/vs_other_frameworks.html) of specific implementation examples using ContextGem and other popular open-source LLM frameworks.


## 📦 Installation

```bash
pip install -U contextgem
```


## 🚀 Quick start

```python
{{QUICK_START}}
```

See more examples in the documentation:

### Basic usage examples
- [Aspect Extraction from Document](https://contextgem.dev/quickstart.html#aspect-extraction-from-document)
- [Extracting Aspect with Sub-Aspects](https://contextgem.dev/quickstart.html#extracting-aspect-with-sub-aspects)
- [Concept Extraction from Aspect](https://contextgem.dev/quickstart.html#concept-extraction-from-aspect)
- [Concept Extraction from Document (text)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-text)
- [Concept Extraction from Document (vision)](https://contextgem.dev/quickstart.html#concept-extraction-from-document-vision)

### Advanced usage examples
- [Extracting Aspects Containing Concepts](https://contextgem.dev/advanced_usage.html#extracting-aspects-with-concepts)
- [Extracting Aspects and Concepts from a Document](https://contextgem.dev/advanced_usage.html#extracting-aspects-and-concepts-from-a-document)
- [Using a Multi-LLM Pipeline to Extract Data from Several Documents](https://contextgem.dev/advanced_usage.html#using-a-multi-llm-pipeline-to-extract-data-from-several-documents)


## 🎯 Focused document analysis

ContextGem leverages LLMs' long context windows to deliver superior extraction accuracy from individual documents. Unlike RAG approaches that often [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f), ContextGem capitalizes on [continuously expanding context capacity](https://arxiv.org/abs/2502.12962), evolving LLM capabilities, and decreasing costs. This focused approach enables direct information extraction from complete documents, eliminating retrieval inconsistencies while optimizing for in-depth single-document analysis. While this delivers maximum accuracy for individual documents, ContextGem does not currently support cross-document querying or corpus-wide retrieval - for these use cases, traditional RAG systems (e.g., LlamaIndex, Haystack) remain more appropriate.

Read more on [how it works](https://contextgem.dev/how_it_works.html) in the documentation.


## ⚡ Optimizations

ContextGem documentation offers guidance on optimization strategies to maximize performance, minimize costs, and enhance extraction accuracy:

- [Optimizing for Accuracy](https://contextgem.dev/optimizations/optimization_accuracy.html)
- [Optimizing for Speed](https://contextgem.dev/optimizations/optimization_speed.html)
- [Optimizing for Cost](https://contextgem.dev/optimizations/optimization_cost.html)
- [Dealing with Long Documents](https://contextgem.dev/optimizations/optimization_long_docs.html)
- [Choosing the Right LLM(s)](https://contextgem.dev/optimizations/optimization_choosing_llm.html)


## 📚 Documentation

Full documentation is available at [contextgem.dev](https://contextgem.dev).

A raw text version of the full documentation is available at [`docs/docs-raw-for-llm.txt`](https://github.com/shcherbak-ai/contextgem/blob/main/docs/docs-raw-for-llm.txt). This file is automatically generated and contains all documentation in a format optimized for LLM ingestion (e.g. for Q&A).


## 💬 Community

If you have a feature request or a bug report, feel free to [open an issue](https://github.com/shcherbak-ai/contextgem/issues/new) on GitHub. If you'd like to discuss a topic or get general advice on using ContextGem for your project, start a thread in [GitHub Discussions](https://github.com/shcherbak-ai/contextgem/discussions/new/).


## 🤝 Contributing

We welcome contributions from the community - whether it's fixing a typo or developing a completely new feature! To get started, please check out our [Contributor Guidelines](https://github.com/shcherbak-ai/contextgem/blob/main/CONTRIBUTING.md).


## 🗺️ Roadmap

ContextGem is at an early stage. Our development roadmap includes:

- **Enhanced Analytical Abstractions**: Building more sophisticated analytical layers on top of the core extraction workflow to enable deeper insights and more complex document understanding
- **API Simplification**: Continuing to refine and streamline the API surface to make document analysis more intuitive and accessible
- **Terminology Refinement**: Improving consistency and clarity of terminology throughout the framework to enhance developer experience

We are committed to making ContextGem the most effective tool for extracting structured information from documents.


## 📄 License & Contact

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/shcherbak-ai/contextgem/blob/main/LICENSE) and [NOTICE](https://github.com/shcherbak-ai/contextgem/blob/main/NOTICE) files for details.

Copyright © 2025 [Shcherbak AI AS](https://shcherbak.ai) - AI engineering company developing tools for AI/ML/NLP developers.

[Connect with us on LinkedIn](https://www.linkedin.com/in/sergii-shcherbak-10068866/) for questions or collaboration ideas.

Built with ❤️ in Oslo, Norway.
