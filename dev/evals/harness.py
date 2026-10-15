# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Shared parts of the evaluation harness: the evaluation cases, the models that
extract them, and how a result is written.

An evaluation case is a document with a reference answer for everything that is
extracted from it: the text (``cases/<case>.txt``), the extraction pipeline
(``cases/<case>.py``), the reference answers (``cases/<case>.gold.json``), and
the paragraphs that are not scored for an aspect (``cases/<case>.neutral.json``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import dotenv_values

from contextgem import (
    BooleanConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    ExtractionPipeline,
    LabelConcept,
    ModelPricing,
    RatingConcept,
)
from dev.evals.cases.complex_dpa import build_complex_dpa_pipeline
from dev.evals.cases.complex_mcu import build_complex_mcu_pipeline
from dev.evals.cases.complex_nda import build_complex_nda_pipeline


EVALS_DIR = Path(__file__).parent
CASES_DIR = EVALS_DIR / "cases"
RESULTS_DIR = EVALS_DIR / "results"
REPO_DIR = EVALS_DIR.parent.parent
PROMPTS_DIR = REPO_DIR / "contextgem" / "internal" / "prompts"
SYSTEM_MESSAGE_DIR = REPO_DIR / "contextgem" / "internal" / "system"

# Kind of the model that extracts a case
ModelKind = Literal["llm", "decision"]

# The function that builds the extraction pipeline of each case
_PIPELINE_BUILDERS: dict[str, Callable[[], ExtractionPipeline]] = {
    "complex_nda": build_complex_nda_pipeline,
    "complex_dpa": build_complex_dpa_pipeline,
    "complex_mcu": build_complex_mcu_pipeline,
}
CASE_NAMES = tuple(_PIPELINE_BUILDERS)

# Concept types that a decision model extracts
_DECISION_CONCEPT_TYPES = (BooleanConcept, LabelConcept, RatingConcept)

# Environment variable with the API key of each hosted provider. The variables
# are read from the environment and from the `.env` file of the repo.
_API_KEY_ENV_VARS = {
    "openai": "CONTEXTGEM_OPENAI_API_KEY",
    "typesafe": "CONTEXTGEM_TYPESAFE_API_KEY",
    "azure_ai": "CONTEXTGEM_AZURE_AI_API_KEY",
}
# Environment variable with the endpoint of the Microsoft Foundry resource
_FOUNDRY_API_BASE_ENV_VAR = "CONTEXTGEM_AZURE_AI_API_BASE"

# Price of 1M input tokens of the hosted decision models, in USD, as of October
# 2026 (output tokens are free). The prices of the LLMs are looked up automatically.
_DECISION_INPUT_PRICES = {"openai": 0.10, "typesafe": 0.042, "azure_ai": 0.042}

# What the host of a Microsoft Foundry resource is replaced with in what is
# written or printed, as it has the name of the resource
_PRIVATE_TEXT_PLACEHOLDER = "<foundry-resource>"


def _keep_for_decision_model(
    concepts: list[Any], gold_concepts: list[dict[str, Any]]
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Keeps the concepts that a decision model extracts (Boolean, label and rating
    concepts), without justifications and references, which a decision model does
    not give, together with the reference answers of the same concepts.

    :param concepts: The concepts of the document or of an aspect.
    :type concepts: list[Any]
    :param gold_concepts: The reference answers of these concepts, in the same order.
    :type gold_concepts: list[dict[str, Any]]
    :return: The concepts kept, and their reference answers without references.
    :rtype: tuple[list[Any], list[dict[str, Any]]]
    """
    kept: list[Any] = []
    kept_gold: list[dict[str, Any]] = []
    for concept, gold_concept in zip(concepts, gold_concepts, strict=True):
        if not isinstance(concept, _DECISION_CONCEPT_TYPES):
            continue
        concept.add_justifications = False
        concept.add_references = False
        kept.append(concept)
        kept_gold.append(
            {
                **gold_concept,
                "items": [
                    {**item, "reference_paragraphs": []}
                    for item in gold_concept["items"]
                ],
            }
        )
    return kept, kept_gold


def _add_neutral_paragraphs(case: str, gold: dict[str, Any]) -> None:
    """
    Adds to the reference answers of each aspect of a case the paragraphs that are
    not scored for the aspect, whether a model extracts them or not, as
    ``neutral_paragraphs``. They are read from ``cases/<case>.neutral.json``:

    - ``headings``: the section headings of the document. A heading is not scored
      for an aspect if its section, i.e. the paragraphs after it up to the next
      heading, has a paragraph of the reference answers of that aspect. A heading
      has no rule or fact of its own: a model that extracts it with its section,
      or that leaves it out, is not wrong.
    - ``borderline``: by the name of an aspect, the paragraphs that careful readers
      disagree on, each with the reason.
    - ``linked``: by the name of an aspect, the paragraphs that give the aspect
      its context without being a part of it, each with the reason: a paragraph
      that a paragraph of the aspect refers to or depends on, or one that
      refers to a clause of the aspect. Extracting such a paragraph gives
      the fuller picture, and leaving it out keeps to the aspect: neither is
      wrong.

    A case without the file has no such paragraphs.

    :param case: The name of the case.
    :type case: str
    :param gold: The reference answers of the document of the case, which are
        changed in place.
    :type gold: dict[str, Any]
    :return: None
    :rtype: None
    :raises ValueError: If a heading is among the paragraphs of the reference
        answers of an aspect, or if an aspect with borderline or linked
        paragraphs is not an aspect of the case.
    """
    path = CASES_DIR / f"{case}.neutral.json"
    neutral: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"headings": []}
    )
    headings = sorted(neutral["headings"])
    sections = {
        heading: set(range(heading + 1, next_heading))
        for heading, next_heading in zip(
            headings, [*headings[1:], len(gold["paragraphs"])], strict=True
        )
    }
    # By the name of an aspect, the paragraphs listed for it with a reason
    listed: dict[str, set[int]] = {}
    for key in ("borderline", "linked"):
        for aspect_name, entries in neutral.get(key, {}).items():
            listed.setdefault(aspect_name, set()).update(
                entry["paragraph"] for entry in entries
            )
    unknown_aspects = set(listed) - {aspect["name"] for aspect in gold["aspects"]}
    if unknown_aspects:
        raise ValueError(
            f"`{path.name}` has paragraphs for aspects that the case `{case}` "
            f"does not have: {sorted(unknown_aspects)}"
        )
    for aspect in gold["aspects"]:
        reference = set(aspect["reference_paragraphs"])
        if reference & set(headings):
            raise ValueError(
                f"The reference answers of the aspect `{aspect['name']}` of "
                f"the case `{case}` have headings: {sorted(reference & set(headings))}"
            )
        aspect["neutral_paragraphs"] = sorted(
            {heading for heading, section in sections.items() if section & reference}
            | listed.get(aspect["name"], set())
        )


def load_case(
    case: str, for_decision_model: bool = False
) -> tuple[Document, dict[str, Any]]:
    """
    Builds the document of an evaluation case, with its extraction pipeline
    assigned and nothing extracted, and loads the reference answers of the case,
    with the paragraphs that are not scored for each aspect added to them (see
    :func:`_add_neutral_paragraphs`).

    :param case: The name of the case, one of ``CASE_NAMES``.
    :type case: str
    :param for_decision_model: Whether the case is to be extracted by a decision
        model: the aspects are kept as they are, and of the concepts, only those
        that a decision model extracts, without justifications and references.
        Defaults to False.
    :type for_decision_model: bool
    :return: The document, and the reference answers of the document (the value of
        ``document`` of the gold file), for the same aspects and concepts.
    :rtype: tuple[Document, dict[str, Any]]
    :raises ValueError: If the paragraphs of the document differ from those of
        the reference answers.
    """
    pipeline = _PIPELINE_BUILDERS[case]()
    text = (CASES_DIR / f"{case}.txt").read_text(encoding="utf-8")
    gold = json.loads((CASES_DIR / f"{case}.gold.json").read_text(encoding="utf-8"))[
        "document"
    ]
    if for_decision_model:
        pipeline.concepts, gold["concepts"] = _keep_for_decision_model(
            list(pipeline.concepts), gold["concepts"]
        )
        for aspect, gold_aspect in zip(pipeline.aspects, gold["aspects"], strict=True):
            aspect.concepts, gold_aspect["concepts"] = _keep_for_decision_model(
                list(aspect.concepts), gold_aspect["concepts"]
            )
    document = Document(raw_text=text, paragraph_segmentation_mode="newlines")
    document.assign_pipeline(pipeline)
    # The reference answers refer to the paragraphs by their positions
    if [paragraph.raw_text for paragraph in document.paragraphs] != [
        paragraph["raw_text"] for paragraph in gold["paragraphs"]
    ]:
        raise ValueError(
            f"The paragraphs of the document of the case `{case}` differ from "
            f"those of its reference answers."
        )
    _add_neutral_paragraphs(case, gold)
    return document, gold


def _get_env_var(name: str) -> str:
    """
    Returns the value of an environment variable, read from the environment or,
    if it is not set there, from the `.env` file of the repo.

    :param name: The name of the variable.
    :type name: str
    :return: The value of the variable.
    :rtype: str
    :raises ValueError: If the variable is not set.
    """
    value = os.environ.get(name) or dotenv_values(REPO_DIR / ".env").get(name)
    if not value:
        raise ValueError(
            f"`{name}` is not set in the environment or in the `.env` file of the repo."
        )
    return value


def create_model(
    model: str, kind: ModelKind, api_base: str | None = None
) -> tuple[DocumentLLM | DocumentDecisionModel, list[str]]:
    """
    Creates the model that extracts a case.

    A model of a hosted provider gets its API key from the environment or from
    the `.env` file of the repo, and prices for its costs: looked up automatically
    for an LLM, and set for a decision model. A model of a local server, i.e. one
    given with an API base, gets no key and no prices.

    :param model: The model identifier, e.g. ``openai/gpt-6-luna``,
        ``typesafe/jev-latest``, ``azure_ai/<deployment name>``,
        ``ollama_chat/mistral-small:24b`` or ``strands_decider/nimble``.
    :type model: str
    :param kind: The kind of the model: "llm" or "decision".
    :type kind: Literal["llm", "decision"]
    :param api_base: The address of a local server, for a model served by it.
        Defaults to None, for a model of a hosted provider.
    :type api_base: str | None
    :return: The model, and the texts that are not to be written or printed (see
        :func:`scrub`): the host of a Microsoft Foundry resource and its name.
    :rtype: tuple[DocumentLLM | DocumentDecisionModel, list[str]]
    :raises ValueError: If the provider of a hosted model is not one that
        the harness has the API key variable of, or if a variable is not set.
    """
    kwargs: dict[str, Any] = {"model": model}
    private_texts: list[str] = []
    provider = model.split("/", 1)[0]
    is_local = api_base is not None
    if is_local:
        kwargs["api_base"] = api_base
    else:
        if provider not in _API_KEY_ENV_VARS:
            raise ValueError(
                f"No API key variable is known for the model `{model}`. Hosted "
                f"providers: {sorted(_API_KEY_ENV_VARS)}. For a model of a local "
                f"server, give the API base."
            )
        kwargs["api_key"] = _get_env_var(_API_KEY_ENV_VARS[provider])
        if provider == "azure_ai":
            # A model deployed in Microsoft Foundry is requested by the name of
            # its deployment, at the endpoint of the resource
            kwargs["api_base"] = _get_env_var(_FOUNDRY_API_BASE_ENV_VAR)
            host = urlparse(kwargs["api_base"]).hostname or ""
            private_texts = [text for text in (host, host.split(".")[0]) if text]
    if kind == "decision":
        if not is_local:
            kwargs["pricing_details"] = ModelPricing(
                input_per_1m_tokens=_DECISION_INPUT_PRICES[provider],
                output_per_1m_tokens=0.0,
            )
        return DocumentDecisionModel(**kwargs), private_texts
    if not is_local:
        kwargs.update(auto_pricing=True, auto_pricing_refresh=False)
    return DocumentLLM(**kwargs), private_texts


def scrub(text: str, private_texts: list[str]) -> str:
    """
    Replaces the texts that are not to be written to a result or printed, e.g.
    in the message of an error of a request.

    :param text: The text to write or print.
    :type text: str
    :param private_texts: The texts to replace (see :func:`create_model`).
    :type private_texts: list[str]
    :return: The text without them.
    :rtype: str
    """
    for private_text in private_texts:
        text = text.replace(private_text, _PRIVATE_TEXT_PLACEHOLDER)
    return text


def get_provenance() -> dict[str, Any]:
    """
    Returns what tells which prompts and which version of ``litellm`` a result is
    recorded with: a short hash of each prompt template of the library, including
    the template of the default system message, so that the results of a changed
    prompt are told apart from the others.

    :return: The version of ``litellm``, and the first 12 characters of
        the SHA-256 hash of each prompt template, by the name of its file.
    :rtype: dict[str, Any]
    """
    prompts = {
        path.name: hashlib.sha256(
            # Whatever the line endings of the checkout are
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()[:12]
        for path in sorted(
            [*PROMPTS_DIR.glob("*.j2"), *SYSTEM_MESSAGE_DIR.glob("*.j2")],
            key=lambda path: path.name,
        )
    }
    return {"litellm": metadata.version("litellm"), "prompts": prompts}


def get_usage_record(model: DocumentLLM | DocumentDecisionModel) -> dict[str, Any]:
    """
    Returns the usage of a model as a result records it: the requests, the tokens
    and the cost.

    :param model: The model, with the requests of the result recorded in its usage.
    :type model: DocumentLLM | DocumentDecisionModel
    :return: The number of requests and of those without a response, the input,
        output and cache tokens, and the cost in USD as a string, which is 0 for
        a model without prices, i.e. a local one.
    :rtype: dict[str, Any]
    """
    usage = model.get_usage()[0].usage
    cost = model.get_cost()[0].cost
    return {
        "requests": len(usage.calls),
        "requests_without_response": sum(
            call.timestamp_received is None for call in usage.calls
        ),
        "input_tokens": usage.input,
        "output_tokens": usage.output,
        "cache_read_tokens": usage.cache_read,
        "cache_write_tokens": usage.cache_write,
        "cost_total": str(cost.total),
    }


def get_model_slug(model: str) -> str:
    """
    Returns a model identifier in a form that can be a part of a file name.

    :param model: The model identifier.
    :type model: str
    :return: The identifier with each run of other characters than letters,
        digits, dots and hyphens replaced with an underscore.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model)


def write_json(path: Path, data: Any) -> None:
    """
    Writes data to a JSON file, creating the folder of the file if needed.

    :param path: The path of the file.
    :type path: Path
    :param data: The data to write.
    :type data: Any
    :return: None
    :rtype: None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
