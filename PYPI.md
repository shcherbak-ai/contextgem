# ContextGem

Effortless LLM extraction from documents. ContextGem is a free, open-source LLM framework that makes it radically easier to extract structured data and insights from documents — with minimal code. You describe what to extract, and ContextGem handles the prompts, the data models, the source references and the justifications.

**Upgrading from 0.x?** ContextGem 1.0 renames and removes classes and parameters, changes some defaults and drops Python 3.10, so code and serialized data written for 0.x may need changes. Follow the [migration guide](https://contextgem.shcherbak.ai/migration_v1/).

```bash
uv add contextgem
```

Or, with pip: `pip install -U contextgem`.

- Generates the prompts from your descriptions of what to extract: there are no prompts to write.
- Returns typed, validated values: text, numbers, dates, booleans, labels, ratings and JSON objects.
- Maps each extracted item back to the paragraphs or sentences of the document that it came from, and can give the reasoning behind it.
- Extracts in context: aspects, sub-aspects and concepts focus the model on the text that matters.
- Runs as one declarative pipeline, reusable and serializable, with model routing, fallbacks and cost tracking.
- Uses the right model for each question: cloud and local LLMs through LiteLLM where it takes reasoning, and decision models where a fast, low-cost answer is enough.
- Gives the probability of each possible answer of a decision model, so that you set the threshold.
- Is built for in-depth analysis of individual documents, not for cross-document querying or corpus-wide retrieval.

See the [documentation](https://contextgem.shcherbak.ai) for the [quickstart](https://contextgem.shcherbak.ai/quickstart/), [how it works](https://contextgem.shcherbak.ai/how_it_works/) and [decision models](https://contextgem.shcherbak.ai/decision_models/decision_models/), the [repository](https://github.com/shcherbak-ai/contextgem#readme) for the overview and the source, and the [changelog](https://github.com/shcherbak-ai/contextgem/blob/main/CHANGELOG.md) for the version history.

Licensed under [Apache-2.0](https://github.com/shcherbak-ai/contextgem/blob/main/LICENSE).
