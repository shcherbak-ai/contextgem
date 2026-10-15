<!-- markdownlint-disable MD033 -->
<p align="center">
  <a href="https://contextgem.shcherbak.ai"><img src="docs/source/_static/contextgem_wordmark.svg" alt="ContextGem" width="576"></a>
</p>

<h1 align="center">Extract structured data and insights from documents<br>using LLMs and decision models</h1>

<p align="center">
  Describe, don't prompt.
</p>

<p align="center">
  <a href="https://pypi.org/project/contextgem/"><img src="https://img.shields.io/pypi/v/contextgem?style=flat-square&labelColor=18181b&color=0e74bc" alt="PyPI version"></a>
  <a href="https://pypi.org/project/contextgem/"><img src="https://img.shields.io/pypi/pyversions/contextgem?style=flat-square&labelColor=18181b&color=0e74bc" alt="Python versions"></a>
  <a href="https://pypi.org/project/contextgem/"><img src="https://img.shields.io/pypi/dm/contextgem?style=flat-square&labelColor=18181b&color=0e74bc&label=downloads" alt="Downloads per month"></a>
  <a href="https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/shcherbak-ai/contextgem/ci-tests.yml?branch=main&style=flat-square&labelColor=18181b&label=CI" alt="CI"></a>
  <a href="https://github.com/shcherbak-ai/contextgem/actions/workflows/ci-tests.yml"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SergiiShcherbak/daaee00e1dfff7a29ca10a922ec3becd/raw/coverage.json&style=flat-square&labelColor=18181b&color=0e74bc" alt="Coverage"></a>
</p>

<p align="center">
  <a href="https://contextgem.shcherbak.ai/quickstart/"><strong>Quickstart</strong></a> ·
  <a href="https://contextgem.shcherbak.ai/how_it_works/"><strong>How it works</strong></a> ·
  <a href="CHANGELOG.md"><strong>Changelog</strong></a> ·
  <a href="https://contextgem.shcherbak.ai/"><strong>Documentation</strong></a>
</p>

<p align="center">
  <img src="docs/source/_static/contextgem_readme_extraction.svg" alt="You write a concept in plain language and make one call. You get every extracted fact with the line of the document it was taken from." width="832"><br>
  <sup>An illustrative contract. Every extracted fact is traced back to its source.</sup>
</p>
<!-- markdownlint-enable MD033 -->

> [!NOTE]
> **New in 1.0: decision models.** Next to LLMs, ContextGem now works with decision models, which answer closed questions (yes/no, one of your labels, a level of a scale) with the probability of each possible answer, fast and at a low cost. A model group sends each aspect and concept to the model that suits it, and any model can fall back on others: see [LLMs and decision models](#llms-and-decision-models).
>
> **Upgrading from 0.x?** Some classes and parameters are renamed or removed, some defaults have changed, and Python 3.10 is no longer supported. Follow the [migration guide](https://contextgem.shcherbak.ai/migration_v1/), or point a coding agent at [llms.txt](https://contextgem.shcherbak.ai/llms.txt), the index of the documentation.

ContextGem is a free, open-source framework that does the work around the models for you: the prompts, the data models, the source references and the justifications, with minimal code.

<!-- markdownlint-disable MD033 -->
<p align="center"><img src="assets/contextgem-separator.svg" alt="" width="456" height="26"></p>
<!-- markdownlint-enable MD033 -->

## Why ContextGem

Reliable extraction from documents usually means writing prompts, designing validation models, mapping outputs back to the source, orchestrating multi-step pipelines and tracking usage across models. ContextGem has all of this built in, in one declarative pipeline that you can reuse and serialize.

<!-- markdownlint-disable MD033 -->
<table>
<tr>
<td width="33%" valign="top"><img src="assets/icons/prompts.svg" alt="" width="32" height="32"><br><strong>No prompts to write</strong><br>Prompts are generated from your descriptions of what to extract.</td>
<td width="33%" valign="top"><img src="assets/icons/typed-output.svg" alt="" width="32" height="32"><br><strong>Typed, validated output</strong><br>Text, numbers, dates, booleans, labels, ratings and JSON objects.</td>
<td width="34%" valign="top"><img src="assets/icons/source.svg" alt="" width="32" height="32"><br><strong>Every fact traced to its source</strong><br>Items map back to their paragraphs or sentences, with the reasoning on request.</td>
</tr>
<tr>
<td valign="top"><img src="assets/icons/nested-context.svg" alt="" width="32" height="32"><br><strong>Nested context</strong><br>Aspects, sub-aspects and concepts focus the model on the text that matters.</td>
<td valign="top"><img src="assets/icons/routing.svg" alt="" width="32" height="32"><br><strong>The right model for each question</strong><br>An LLM where it takes reasoning, a decision model where a fast, low-cost answer is enough.</td>
<td valign="top"><img src="assets/icons/decision-model.svg" alt="" width="32" height="32"><br><strong>Answers with probabilities</strong><br>A decision model gives the probability of each possible answer, and you set the threshold.</td>
</tr>
</table>
<!-- markdownlint-enable MD033 -->

Read more on the [motivation](https://contextgem.shcherbak.ai/motivation/) behind ContextGem.

## LLMs and decision models

ContextGem works with two kinds of models. You describe what to extract in the same way for both, and each aspect and concept goes to the model that suits it.

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/source/_static/contextgem_readme_models.svg" alt="A model group with an LLM and a decision model. The open question, the notice period, goes to the LLM, which writes the answer: 30 days. Two closed questions go to the decision model, which gives the probability of each possible answer: termination for convenience is True with 0.98, above the decision threshold, and the notice form is Written with 1.00, the most probable of its labels." width="832"><br>
  <sup>An illustrative service agreement: the open question goes to the LLM, the closed ones to the decision model.</sup>
</p>
<!-- markdownlint-enable MD033 -->

<!-- markdownlint-disable MD033 -->
| | <img src="assets/icons/llm.svg" alt="" width="32" height="32"><br>LLM | <img src="assets/icons/decision-model.svg" alt="" width="32" height="32"><br>Decision model |
|---|---|---|
| **How it answers** | Writes the answer, which ContextGem validates into typed items | Gives the probability of each possible answer, and writes no text |
| **What it extracts** | Aspects and every type of concept, from text and images | Aspects, and Boolean, label and rating concepts, from text |
| **With each item** | References to paragraphs or sentences, and a justification on request | The probabilities of all the possible answers, and a decision threshold that you set for the model, or for each aspect and concept |
| **Use it for** | Open values, and answers that take reasoning | Closed questions in volume, where a fast, low-cost answer is enough |
| **Models** | Any cloud or local LLM through [LiteLLM](https://github.com/BerriAI/litellm): OpenAI, Anthropic, Google, Azure OpenAI, xAI, Ollama, LM Studio and more | TypeSafe Jev, OpenAI Decisions API, Microsoft Foundry and more, cloud or local |
| **Read more** | [Supported LLMs](https://contextgem.shcherbak.ai/llms/supported_llms/), [configuration](https://contextgem.shcherbak.ai/llms/llm_config/), [extraction methods](https://contextgem.shcherbak.ai/llms/llm_extraction_methods/) | [Decision models](https://contextgem.shcherbak.ai/decision_models/decision_models/) |
<!-- markdownlint-enable MD033 -->

> **Choosing a model.** ContextGem is tested with hosted LLMs such as `gpt-6-luna` (OpenAI) and `claude-haiku-5-5` (Anthropic), and with decision models such as Jev (TypeSafe). A smaller LLM, such as many that run locally, may not follow the detailed extraction instructions: check it on a sample of your documents, and see the [troubleshooting guide](https://contextgem.shcherbak.ai/optimizations/optimization_small_llm_troubleshooting/) if it falls short. Before you give questions to a decision model, compare it with an LLM on a sample of your documents: see [when to use a decision model](https://contextgem.shcherbak.ai/decision_models/decision_models/#when-to-use-a-decision-model).

<!-- markdownlint-disable MD033 -->
<p align="center"><img src="assets/contextgem-separator.svg" alt="" width="456" height="26"></p>
<!-- markdownlint-enable MD033 -->

## Quick start

```bash
uv add contextgem  # or: pip install -U contextgem
```

### Extract with an LLM

An LLM extracts aspects and values of any type. Below, it extracts two things from a contract: an aspect, the **payment terms**, which comes back as the paragraphs on payments, and a concept, the **anomalies**, which span the whole document and which retrieval-based approaches can miss. Each anomaly comes with a justification and the sentence it was found in.

```python
import os

from contextgem import Aspect, Document, DocumentLLM, StringConcept


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

doc.add_aspects(
    [
        Aspect(
            name="Payment terms",
            description="Clauses on payments, fees and invoices",
        )
    ]
)
doc.add_concepts(
    [
        StringConcept(
            name="Anomalies",
            description="Anomalies in the document",
            add_references=True,
            reference_depth="sentences",
            add_justifications=True,
            justification_depth="brief",
        )
    ]
)

llm = DocumentLLM(
    model="openai/gpt-6-luna",  # or another LLM, cloud or local
    api_key=os.environ.get("CONTEXTGEM_OPENAI_API_KEY"),
)
doc = llm.extract_all(doc)

payment_terms = doc.get_aspect_by_name("Payment terms").extracted_items
# 1 item, the paragraph: "The Customer shall pay the Supplier within 30 calendar
#     days of receiving an invoice..."

anomalies = doc.get_concept_by_name("Anomalies").extracted_items
# 2 items. The first one:
# .value: "The statement that a purple elephant danced on the moon while eating
#     ice cream is anomalous and unrelated to the consultancy agreement."
# .justification: "It is an implausible, fantastical statement in an otherwise
#     contractual document."
# .reference_sentences: the sentence of the purple elephant

```

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/source/_static/contextgem_readme_aspects_concepts.svg" alt="What the example above extracts from its contract. The aspect Payment terms gives back the paragraph on payments as it is written in the document. The concept Anomalies gives two new values, each with a justification and the sentence it was found in: the purple elephant that danced on the moon, and the time-traveling dinosaurs that will review the deliverables." width="832"><br>
  <sup>What the example above extracts: an aspect gives back the document's own text, a concept gives new values with their sources.</sup>
</p>
<!-- markdownlint-enable MD033 -->

### Decide with a decision model

A decision model answers closed questions (yes/no, one of your labels, a level of a scale) with the probability of each possible answer. Below, it extracts an aspect from a contract, the **termination** clauses, by deciding for each paragraph whether it is one, and answers two questions about the contract, concepts, in one request.

```python
import os

from contextgem import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentDecisionModel,
    LabelConcept,
)


doc = Document(
    raw_text=(
        "Service Agreement\n"
        "Either party may terminate this Agreement for any reason by giving 30 days' "
        "written notice to the other party.\n"
        "This Agreement is governed by the laws of the State of New York.\n"
    ),
)

doc.add_aspects(
    [
        Aspect(
            name="Termination",
            description="Clauses on termination of the agreement",
        )
    ]
)
doc.add_concepts(
    [
        BooleanConcept(
            name="Termination for convenience",
            description="Either party may terminate the agreement without cause",
        ),
        LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=["Written", "Email", "Any form", "Not specified"],
            classification_type="multi_class",
        ),
    ]
)

decision_model = DocumentDecisionModel(
    model="typesafe/jev-latest",  # or another decision model, cloud or local
    api_key=os.environ.get("CONTEXTGEM_TYPESAFE_API_KEY"),
)
# The aspect: a yes/no question for each paragraph. The concepts: both in one request.
doc = decision_model.extract_all(doc)

termination_clauses = doc.get_aspect_by_name("Termination").extracted_items
# 1 item, the paragraph: "Either party may terminate this Agreement for any reason..."
# .probabilities: {True: 0.99, False: 0.01} (rounded); the other two paragraphs
#     had 0.26 and 0.03, below the decision threshold of 0.5

convenience = doc.get_concept_by_name("Termination for convenience").extracted_items[0]
# .value: True
# .probabilities: {True: 0.98, False: 0.02} (rounded)

notice_form = doc.get_concept_by_name("Notice form").extracted_items[0]
# .value: ["Written"]
# .probabilities: {"Written": 1.0, "Email": 0.0, "Any form": 0.0, "Not specified": 0.0}

```

More examples: [aspects](https://contextgem.shcherbak.ai/quickstart/#aspect-extraction-from-document), [sub-aspects](https://contextgem.shcherbak.ai/quickstart/#extracting-aspect-with-sub-aspects), [concepts of an aspect](https://contextgem.shcherbak.ai/quickstart/#concept-extraction-from-aspect), [concepts from images](https://contextgem.shcherbak.ai/quickstart/#concept-extraction-from-document-vision), [aspects containing concepts](https://contextgem.shcherbak.ai/advanced_usage/#extracting-aspects-containing-concepts), [aspects with a decision model](https://contextgem.shcherbak.ai/decision_models/decision_models/#extracting-aspects), [a decision model and an LLM in one model group](https://contextgem.shcherbak.ai/decision_models/decision_models/#using-a-decision-model-in-a-model-group), [multi-model pipelines](https://contextgem.shcherbak.ai/advanced_usage/#using-a-multi-llm-pipeline-to-extract-data-from-several-documents) and a [chat interface with tool calling](https://contextgem.shcherbak.ai/quickstart/#lightweight-llm-chat-interface).

<!-- markdownlint-disable MD033 -->
<p align="center"><img src="assets/contextgem-separator.svg" alt="" width="456" height="26"></p>
<!-- markdownlint-enable MD033 -->

## How it works

<!-- markdownlint-disable MD033 -->
<p align="center">
  <img src="docs/source/_static/contextgem_readme_structure.svg" alt="A document with two aspects and four concepts, the type of each, and what is extracted: paragraphs for an aspect, and a typed value for a concept." width="832"><br>
  <sup>An illustrative contract: what you define, and what is extracted.</sup>
</p>
<!-- markdownlint-enable MD033 -->

1. **Wrap your content in a [Document](https://contextgem.shcherbak.ai/documents/document_config/)**: text, images, or both.
2. **Say what to extract.** [Aspects](https://contextgem.shcherbak.ai/aspects/aspects/) pull out the parts of the document on a topic, as paragraphs or sentences. [Concepts](https://contextgem.shcherbak.ai/concepts/supported_concepts/) extract typed values from the whole document or from an aspect, with optional references and justifications. Keep a set of both as a reusable [extraction pipeline](https://contextgem.shcherbak.ai/pipelines/extraction_pipelines/).
3. **Run a model.** One call to an [LLM](https://contextgem.shcherbak.ai/llms/llm_extraction_methods/), a [decision model](https://contextgem.shcherbak.ai/decision_models/decision_models/) or a [model group](https://contextgem.shcherbak.ai/llms/llm_config/#model-groups) of both extracts everything. A group sends each aspect and concept to the model of its role, e.g. the closed questions to a decision model and the rest to an LLM, and each model can have fallbacks.

Read more on [how ContextGem works](https://contextgem.shcherbak.ai/how_it_works/) in the documentation.

## Every paragraph is read

ContextGem is built for in-depth analysis of individual documents. There is no search step that picks the chunks that look relevant, as in RAG: the model reads every paragraph of the text it extracts from, all at once or in [batches of paragraphs](https://contextgem.shcherbak.ai/optimizations/optimization_long_docs/) of a size you set, so no fact is missed because a search ranked its passage low. This avoids the retrieval inconsistencies that make RAG [struggle with complex concepts and nuanced insights](https://www.linkedin.com/pulse/raging-contracts-pitfalls-rag-contract-review-shcherbak-ai-ptg3f).

The trade-off: ContextGem does not query across documents or search a corpus. For these, RAG frameworks (e.g. LlamaIndex, Haystack) remain more appropriate.

## How it is tested

Besides tests that replay recorded responses of real models, [Hypothesis](https://hypothesis.readthedocs.io/) property tests run whole extractions on generated model outputs, including what a model should never return: IDs that are not in the prompt, values of the wrong type or not valid JSON, and responses cut off at the max output tokens. They check that such an output is rejected and retried, or reported with an error or a warning, and that it never yields extracted items. A change that can affect the quality of the extraction is measured on evaluation cases, i.e. complex documents with reference answers, against a recorded baseline of quality and cost. What exact matching cannot compare (narrative values, justifications and text fields) is judged by a decision model, Jev: it answers one yes/no question at a time with a probability, and is checked on answers whose verdict is known before it is used. See [Testing](CONTRIBUTING.md#-testing).

## Documentation

Full documentation: **[contextgem.shcherbak.ai](https://contextgem.shcherbak.ai)**

<!-- markdownlint-disable MD033 -->
| <img src="assets/icons/get-started.svg" alt="" width="32" height="32"><br>Get started | <img src="assets/icons/nested-context.svg" alt="" width="32" height="32"><br>Define what to extract | <img src="assets/icons/models.svg" alt="" width="32" height="32"><br>Run models | <img src="assets/icons/optimize.svg" alt="" width="32" height="32"><br>Optimize |
|:------------|:-----------------------|:-----------|:---------|
| [Installation](https://contextgem.shcherbak.ai/installation/)<br>[Quickstart](https://contextgem.shcherbak.ai/quickstart/)<br>[Advanced usage](https://contextgem.shcherbak.ai/advanced_usage/)<br>[How it works](https://contextgem.shcherbak.ai/how_it_works/)<br>[Migrating to v1.0](https://contextgem.shcherbak.ai/migration_v1/) | [Documents](https://contextgem.shcherbak.ai/documents/document_config/)<br>[Aspects](https://contextgem.shcherbak.ai/aspects/aspects/)<br>[Concepts](https://contextgem.shcherbak.ai/concepts/supported_concepts/)<br>[Extraction pipelines](https://contextgem.shcherbak.ai/pipelines/extraction_pipelines/)<br>[Serialization](https://contextgem.shcherbak.ai/serialization/) | [Supported LLMs](https://contextgem.shcherbak.ai/llms/supported_llms/)<br>[LLM configuration](https://contextgem.shcherbak.ai/llms/llm_config/)<br>[Extraction methods](https://contextgem.shcherbak.ai/llms/llm_extraction_methods/)<br>[Decision models](https://contextgem.shcherbak.ai/decision_models/decision_models/)<br>[Chat with tools](https://contextgem.shcherbak.ai/llms/chat_with_tools/) | [Accuracy](https://contextgem.shcherbak.ai/optimizations/optimization_accuracy/)<br>[Speed](https://contextgem.shcherbak.ai/optimizations/optimization_speed/)<br>[Cost](https://contextgem.shcherbak.ai/optimizations/optimization_cost/)<br>[Long documents](https://contextgem.shcherbak.ai/optimizations/optimization_long_docs/)<br>[Choosing the right LLM(s)](https://contextgem.shcherbak.ai/optimizations/optimization_choosing_llm/) |
<!-- markdownlint-enable MD033 -->

For LLMs and coding agents, [llms.txt](https://contextgem.shcherbak.ai/llms.txt) indexes the documentation, with a Markdown copy of each page.

For visual architecture maps and natural language Q&A on the codebase, see [DeepWiki](https://deepwiki.com/shcherbak-ai/contextgem).

## Community

<!-- markdownlint-disable MD033 -->
<table>
<tr>
<td width="50%" valign="top"><strong>Need help or want to discuss?</strong><br>Start a thread in <a href="https://github.com/shcherbak-ai/contextgem/discussions/new/">GitHub Discussions</a>.</td>
<td width="50%" valign="top"><strong>Found a bug or have a feature request?</strong><br><a href="https://github.com/shcherbak-ai/contextgem/issues/new">Open an issue</a>.</td>
</tr>
<tr>
<td valign="top"><strong>Want to contribute?</strong><br>Contributions are welcome, from a fixed typo to a new feature: see the <a href="CONTRIBUTING.md">Contributor Guidelines</a>.</td>
<td valign="top"><strong>Security</strong><br>The project is scanned with <a href="https://codeql.github.com/">CodeQL</a> and <a href="https://github.com/PyCQA/bandit">Bandit</a>, and with <a href="https://snyk.io">Snyk</a> as needed: see the <a href="SECURITY.md">security policy</a>.</td>
</tr>
</table>
<!-- markdownlint-enable MD033 -->

## Acknowledgements

ContextGem relies on these excellent open-source packages: [aiolimiter](https://github.com/mjpieters/aiolimiter), [colorlog](https://github.com/borntyping/python-colorlog), [docstring-parser](https://github.com/rr-/docstring_parser), [fastjsonschema](https://github.com/horejsek/python-fastjsonschema), [genai-prices](https://github.com/pydantic/genai-prices), [Jinja2](https://github.com/pallets/jinja), [litellm](https://github.com/BerriAI/litellm), [pillow](https://github.com/python-pillow/Pillow), [pydantic](https://github.com/pydantic/pydantic), [python-ulid](https://github.com/mdomke/python-ulid), [tenacity](https://github.com/jd/tenacity) and [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite).

## License

Apache 2.0 License - see the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.

© 2025 [Shcherbak AI AS](https://shcherbak.ai) — Enterprise AI Engineering. We build AI agents to fit your enterprise — each verified, constrained, and governed.

<!-- markdownlint-disable MD033 -->
<p align="center"><img src="assets/contextgem-separator.svg" alt="" width="456" height="26"></p>
<!-- markdownlint-enable MD033 -->

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
