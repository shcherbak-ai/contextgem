<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://contextgem.dev"><img src="docs/source/_static/contextgem_wordmark.svg" alt="ContextGem" width="576"></a>
</p>

<h1 align="center">Effortless LLM extraction from documents</h1>

<p align="center">
  Minimal code · Automated prompts · Multi-model
</p>

<p align="center">
  <a href="https://pypi.org/project/contextgem/"><img src="https://img.shields.io/pypi/v/contextgem?style=flat-square&labelColor=18181b&color=38bdf8" alt="PyPI version"></a>
  <a href="https://pypi.org/project/contextgem/"><img src="https://img.shields.io/pypi/pyversions/contextgem?style=flat-square&labelColor=18181b&color=38bdf8" alt="Python versions"></a>
  <a href="https://pypi.org/project/contextgem/"><img src="https://img.shields.io/pypi/dm/contextgem?style=flat-square&labelColor=18181b&color=38bdf8&label=downloads" alt="Downloads per month"></a>
  <a href="https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/shcherbak-ai/contextgem/ci-tests.yml?branch=main&style=flat-square&labelColor=18181b&label=CI" alt="CI"></a>
  <a href="https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SergiiShcherbak/daaee00e1dfff7a29ca10a922ec3becd/raw/coverage.json&style=flat-square&labelColor=18181b&color=38bdf8" alt="Coverage"></a>
</p>

<p align="center">
  <a href="https://contextgem.dev/quickstart/"><strong>Quickstart</strong></a> ·
  <a href="https://contextgem.dev/how_it_works/"><strong>How it works</strong></a> ·
  <a href="CHANGELOG.md"><strong>Changelog</strong></a> ·
  <a href="https://contextgem.dev/"><strong>Documentation</strong></a>
</p>

<p align="center">
  <img src="docs/source/_static/contextgem_readme_extraction.svg" alt="You write a concept in plain language and make one call. You get every extracted fact with the line of the document it was taken from." width="832"><br>
  <sup>An illustrative contract. Every extracted fact is traced back to its source.</sup>
</p>
<!-- markdownlint-enable MD033 -->

ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code. You describe *what* to extract, and ContextGem handles *how*: the prompts, the data models, the source references and the justifications.

## Why ContextGem

Reliable extraction from documents usually means writing prompts, designing validation models, mapping outputs back to the source, orchestrating multi-step pipelines and tracking usage across models. ContextGem has all of this built in, in one declarative pipeline that you can reuse and serialize.

<!-- markdownlint-disable MD033 -->
<table>
<tr>
<td width="33%"><strong>No prompts to write</strong><br>Prompts are generated from your descriptions of what to extract.</td>
<td width="33%"><strong>Typed, validated output</strong><br>Text, numbers, dates, booleans, labels, ratings and JSON objects.</td>
<td width="34%"><strong>Every fact traced to its source</strong><br>Items map back to their paragraphs or sentences, with the reasoning on request.</td>
</tr>
<tr>
<td><strong>Nested context</strong><br>Aspects, sub-aspects and concepts focus the model on the text that matters.</td>
<td><strong>The right model for each question</strong><br>An LLM where it takes reasoning, a decision model where a fast, low-cost answer is enough.</td>
<td><strong>Answers with probabilities</strong><br>A decision model gives the probability of each possible answer, and you set the threshold.</td>
</tr>
</table>
<!-- markdownlint-enable MD033 -->

Read more on the [motivation](https://contextgem.dev/motivation/) behind ContextGem.

## Quick start

```bash
uv add contextgem  # or: pip install -U contextgem
```

The example below extracts **anomalies** from a contract: a concept that takes the context of the whole document, which retrieval-based approaches can miss. Each anomaly comes back with a justification and with the sentences it was found in.

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
    model="openai/gpt-6-luna",  # or another provider/LLM
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

More examples: [aspects](https://contextgem.dev/quickstart/#aspect-extraction-from-document), [sub-aspects](https://contextgem.dev/quickstart/#extracting-aspect-with-sub-aspects), [concepts of an aspect](https://contextgem.dev/quickstart/#concept-extraction-from-aspect), [concepts from images](https://contextgem.dev/quickstart/#concept-extraction-from-document-vision), [aspects containing concepts](https://contextgem.dev/advanced_usage/#extracting-aspects-containing-concepts), [multi-model pipelines](https://contextgem.dev/advanced_usage/#using-a-multi-llm-pipeline-to-extract-data-from-several-documents) and a [chat interface with tool calling](https://contextgem.dev/quickstart/#lightweight-llm-chat-interface).

## How it works

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/source/_static/contextgem_readme_structure.svg" alt="A document with two aspects and four concepts, the type of each, and what is extracted: paragraphs for an aspect, and a typed value for a concept." width="832"><br>
  <sup>An illustrative contract: what you define, and what is extracted.</sup>
</p>
<!-- markdownlint-enable MD033 -->

1. **Wrap your content in a [Document](https://contextgem.dev/documents/document_config/)**: text, images, or both.
2. **Say what to extract.** [Aspects](https://contextgem.dev/aspects/aspects/) pull out the parts of the document on a topic, as paragraphs or sentences. [Concepts](https://contextgem.dev/concepts/supported_concepts/) extract typed values from the whole document or from an aspect, with optional references and justifications. Keep a set of both as a reusable [extraction pipeline](https://contextgem.dev/pipelines/extraction_pipelines/).
3. **Run a model.** One call to an [LLM](https://contextgem.dev/llms/llm_extraction_methods/) extracts everything. A [model group](https://contextgem.dev/llms/llm_config/#model-groups) routes aspects and concepts to different models by role, e.g. the closed questions to a [decision model](https://contextgem.dev/decision_models/decision_models/), and each model can have fallbacks.

Read more on [how ContextGem works](https://contextgem.dev/how_it_works/) in the documentation.

## Models

ContextGem works with two kinds of models, and routes each aspect and concept to the one that suits it: an LLM where it takes reasoning, a decision model where a fast, low-cost answer with a probability is enough.

**LLMs.** Any cloud or local LLM through [LiteLLM](https://github.com/BerriAI/litellm): OpenAI, Anthropic, Google, Azure OpenAI, xAI, Ollama, LM Studio and more, with or without reasoning capabilities, behind one interface. See the [supported LLMs](https://contextgem.dev/llms/supported_llms/), the [LLM configuration](https://contextgem.dev/llms/llm_config/) and the [extraction methods](https://contextgem.dev/llms/llm_extraction_methods/).

**Decision models.** A decision model (e.g. TypeSafe Jev, OpenAI Decisions API, Microsoft Foundry) answers typed questions about a text (yes/no, pick one option, rate on a scale) with probabilities, and generates no text. It extracts aspects, and Boolean, label and rating concepts, on its own, in a model group next to an LLM, or in a fallback chain. Each extracted item has the probabilities of all the possible answers, and a decision threshold sets what counts as a "yes". Concepts with open values, justifications, references for concepts and extraction from images require an LLM. See [decision models](https://contextgem.dev/decision_models/decision_models/).

> **Choosing a model.** For reliable extraction, use a model at least as capable as `gpt-6-luna`. Smaller models (such as 8B parameter models) may struggle with the detailed extraction instructions: see the [troubleshooting guide](https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting/).

## Whole documents, not fragments

ContextGem is built for in-depth analysis of individual documents. It gives the model the whole document, using the long context windows of LLMs, instead of retrieving fragments of it. This avoids the retrieval inconsistencies that make RAG [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f), and it benefits from growing context windows, more capable LLMs and falling costs.

The trade-off: ContextGem does not support cross-document querying or corpus-wide retrieval. For these, RAG frameworks (e.g. LlamaIndex, Haystack) remain more appropriate.

## Documentation

Full documentation: **[contextgem.dev](https://contextgem.dev)**

| Get started | Define what to extract | Run models | Optimize |
|-------------|------------------------|------------|----------|
| [Installation](https://contextgem.dev/installation/)<br>[Quickstart](https://contextgem.dev/quickstart/)<br>[Advanced usage](https://contextgem.dev/advanced_usage/)<br>[How it works](https://contextgem.dev/how_it_works/)<br>[Migrating to v1.0](https://contextgem.dev/migration_v1/) | [Documents](https://contextgem.dev/documents/document_config/)<br>[Aspects](https://contextgem.dev/aspects/aspects/)<br>[Concepts](https://contextgem.dev/concepts/supported_concepts/)<br>[Extraction pipelines](https://contextgem.dev/pipelines/extraction_pipelines/)<br>[Serialization](https://contextgem.dev/serialization/) | [Supported LLMs](https://contextgem.dev/llms/supported_llms/)<br>[LLM configuration](https://contextgem.dev/llms/llm_config/)<br>[Extraction methods](https://contextgem.dev/llms/llm_extraction_methods/)<br>[Decision models](https://contextgem.dev/decision_models/decision_models/)<br>[Chat with tools](https://contextgem.dev/llms/chat_with_tools/) | [Accuracy](https://contextgem.dev/optimizations/optimization_accuracy/)<br>[Speed](https://contextgem.dev/optimizations/optimization_speed/)<br>[Cost](https://contextgem.dev/optimizations/optimization_cost/)<br>[Long documents](https://contextgem.dev/optimizations/optimization_long_docs/)<br>[Choosing the right LLM(s)](https://contextgem.dev/optimizations/optimization_choosing_llm/) |

For visual architecture maps and natural language Q&A on the codebase, see [DeepWiki](https://deepwiki.com/shcherbak-ai/contextgem).

## Community

- **Need help or want to discuss?** Start a thread in [GitHub Discussions](https://github.com/shcherbak-ai/contextgem/discussions/new/).
- **Found a bug or have a feature request?** [Open an issue](https://github.com/shcherbak-ai/contextgem/issues/new).
- **Want to contribute?** Contributions are welcome, from a fixed typo to a new feature: see the [Contributor Guidelines](CONTRIBUTING.md).
- **Security.** The project is scanned with [CodeQL](https://codeql.github.com/) and [Bandit](https://github.com/PyCQA/bandit), and with [Snyk](https://snyk.io) as needed: see the [security policy](SECURITY.md).

## Acknowledgements

ContextGem relies on these excellent open-source packages: [aiolimiter](https://github.com/mjpieters/aiolimiter), [colorlog](https://github.com/borntyping/python-colorlog), [docstring-parser](https://github.com/rr-/docstring_parser), [fastjsonschema](https://github.com/horejsek/python-fastjsonschema), [genai-prices](https://github.com/pydantic/genai-prices), [Jinja2](https://github.com/pallets/jinja), [litellm](https://github.com/BerriAI/litellm), [pillow](https://github.com/python-pillow/Pillow), [pydantic](https://github.com/pydantic/pydantic), [python-ulid](https://github.com/mdomke/python-ulid), [tenacity](https://github.com/jd/tenacity) and [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite).

## License

Apache 2.0 License - see the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.

© 2025 [Shcherbak AI AS](https://shcherbak.ai) — Enterprise AI Engineering. We build AI agents that transform how enterprises operate.

## More from shcherbak_ai

<!-- markdownlint-disable MD033 -->
<table>
  <tr>
    <td width="50%" align="center" valign="top">
      <a href="https://github.com/shcherbak-ai/licenseal"><img src="assets/licenseal-wordmark.svg" alt="licenseal" width="336"></a><br>
      License compatibility checks for dependency trees, fast enough for CI.
    </td>
    <td width="50%" align="center" valign="top">
      <a href="https://github.com/shcherbak-ai/tethered"><img src="assets/tethered-wordmark.svg" alt="tethered" width="336"></a><br>
      Network egress control for CLI tools and agent workflows.
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

Questions or collaboration ideas? Reach out on [LinkedIn](https://www.linkedin.com/in/sergii-shcherbak-10068866/) or [X](https://x.com/seshch). Built with ❤️ in Oslo, Norway.
