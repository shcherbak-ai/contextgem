"""
Generic benchmarking utilities for ContextGem.

This module provides a flexible runner that can evaluate extraction quality
for any document + targets set defined in an external module. It supports:

- Loading a benchmark set (document + interconnected aspects/concepts) by module name
- Running extraction with a provided LLM and judging with another LLM
- Presence-based and judge-based scoring per category inferred from metadata

Conventions
-----------
- Category attribution defaults to `custom_data["benchmark_category"]` on
  each Aspect/Concept. Example categories used by the built-in contract set:
  "simple_aspect", "reasoning_aspect", "hallucination_aspect",
  "simple_concept", "reasoning_concept", "hallucination_concept".
- Hallucination categories should be configured as must-not-have and typically
  are excluded from judge-based scoring.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass

from contextgem import Aspect, Document, DocumentLLM
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.items import _ExtractedItem
from contextgem.internal.loggers import logger


# Defaults
MIN_SCORE_THRESHOLD: float = 90.0
IDEAL_JUDGE_RATING: int = 5


@dataclass
class _BenchmarkScores:
    """
    Aggregated benchmark scores in percentages.

    Presence-based metrics reflect whether items were extracted (for required
    targets) or not extracted (for hallucination targets). Judge-based metrics
    are derived from a 0-5 rating returned by the judge LLM and normalized to
    a 0-100 percentage per category.
    """

    # Presence-based (must-have/must-not-have) accuracy
    presence_simple_aspects: float
    presence_reasoning_aspects: float
    presence_simple_concepts: float
    presence_reasoning_concepts: float
    presence_hallucination_aspects: float
    presence_hallucination_concepts: float

    # Judge-based quality
    judge_simple_aspects: float
    judge_reasoning_aspects: float
    judge_simple_concepts: float
    judge_reasoning_concepts: float

    # Non-compliance stats
    presence_non_compliant_aspects_must_have_count: int
    presence_non_compliant_concepts_must_have_count: int
    presence_non_compliant_hallucination_aspects_count: int
    presence_non_compliant_hallucination_concepts_count: int
    judge_non_compliant_aspects_count: int
    judge_non_compliant_concepts_count: int

    total: float


def _pct(numerator: int, denominator: int) -> float:
    """
    Calculates the percentage of numerator over denominator, rounded to two decimal places.

    :param numerator: The numerator value.
    :type numerator: int
    :param denominator: The denominator value.
    :type denominator: int
    :return: The percentage value as a float. Returns 0.0 if denominator is zero or negative.
    :rtype: float
    """
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _load_benchmark_document(module_name: str) -> Document:
    """
    Loads a benchmark document + extraction targets (aspects/concepts) by importing a module.

    Expected callable names in the module (in order of preference):
    - build_benchmark_document() -> Document  # Document already has extraction targets added
    """

    mod = importlib.import_module(module_name)
    if hasattr(mod, "build_benchmark_document"):
        doc = mod.build_benchmark_document()
        if not isinstance(doc, Document):
            raise TypeError(
                f"{module_name}.build_benchmark_document() must return a Document"
            )
        if not doc.aspects and not doc.concepts:
            raise ValueError(
                f"{module_name}.build_benchmark_document() must add at least one "
                "aspect or concept to the document"
            )
        return doc

    raise ImportError(f"Module '{module_name}' must define build_benchmark_document()")


def _serialize_item_for_prompt(item: _ExtractedItem) -> dict:
    """
    Serializes an extracted item of an Aspect or _Concept instance for use in a LLM judge prompt,
    converting its attributes to a dictionary.

    :param item: The extracted item to serialize. Expected to have attributes such as
        'value', 'justification', 'reference_paragraphs', and 'reference_sentences'.
    :type item: _ExtractedItem
    :return: A dictionary representation of the extracted item, including type, value,
        justification, and optionally reference paragraphs and sentences if present.
    :rtype: dict
    """
    d: dict = {
        "type": item.__class__.__name__,
        "value": item.value,
        "justification": item.justification,
    }
    ref_paras = [p.raw_text for p in item.reference_paragraphs]
    if ref_paras:
        d["reference_paragraphs"] = ref_paras
    ref_sents = [s.raw_text for s in item.reference_sentences]
    if ref_sents:
        d["reference_sentences"] = ref_sents
    return d


def _items_block(inst: Aspect | _Concept) -> str:
    """
    Formats the extracted items of an Aspect or _Concept instance for rendering
    in a LLM judge prompt.

    Each extracted item is serialized to JSON and wrapped in <extracted_item> tags.
    If no items are present, returns "No items extracted".

    :param inst: The Aspect or _Concept instance whose extracted_items will be formatted.
    :type inst: Aspect | _Concept
    :return: A string containing all extracted items formatted for prompt inclusion.
    :rtype: str
    """
    items = inst.extracted_items
    lines: list[str] = []
    for it in items:
        content = _serialize_item_for_prompt(it)
        content_str = json.dumps(content, ensure_ascii=False)
        lines.append(f"<extracted_item>\n{content_str}\n</extracted_item>")
    if lines:
        return "\n".join(lines)
    return "No items extracted"


def _judge_target(
    judge_llm: DocumentLLM,
    target_type: str,
    name: str,
    items_str: str,
    doc_text: str,
) -> tuple[int, str]:
    """
    Uses the judge LLM to evaluate the quality of extracted items for a given target.

    :param judge_llm: The LLM instance used to judge the extraction quality.
    :type judge_llm: DocumentLLM
    :param target_type: The type of the target being evaluated (e.g., aspect or concept).
    :type target_type: str
    :param name: The name of the target being evaluated.
    :type name: str
    :param items_str: The string representation of the extracted items.
    :type items_str: str
    :param doc_text: The full text of the document from which items were extracted.
    :type doc_text: str
    :return: A tuple containing the integer rating and a justification string.
    :rtype: tuple[int, str]
    """
    prompt = (
        "You are a strict evaluator of quality of data extraction from the document "
        "specified within <document_text></document_text> tags.\n"
        "Rate the correctness and grounding of the extracted items "
        "(specified within <extracted_items></extracted_items> tags) for the target "
        "(specified within <target_type></target_type> and <target_name></target_name> tags) "
        f"on a scale from 0 to {IDEAL_JUDGE_RATING}, where 0 is the worst and {IDEAL_JUDGE_RATING} is the best.\n"
        "Consider factual correctness, support by the provided document text, and completeness.\n"
        "Do not evaluate the schema or structure of the extracted items; "
        "assume the schema is already validated. Focus only on the values of the fields.\n"
        f"If the rating is less than {IDEAL_JUDGE_RATING}, include up to 3 sentences of justification. "
        f"If the rating equals {IDEAL_JUDGE_RATING}, justification must be an empty string.\n"
        "Return ONLY a JSON object with keys 'rating' (integer) and 'justification' (string).\n\n"
        f"<document_text>\n{doc_text}\n</document_text>\n\n"
        f"<target_type>{target_type}</target_type>\n"
        f"<target_name>{name}</target_name>\n"
        f"<extracted_items>\n{items_str}\n</extracted_items>\n"
    )
    try:
        resp = judge_llm.chat(prompt)
        txt = str(resp).strip()
        data = json.loads(txt)
        rating = int(data.get("rating", 0))
        justification = data.get("justification", "").strip()
        return rating, justification
    except Exception:
        return 0, ""


def run_benchmark_for_module(
    llm: DocumentLLM,
    judge_llm: DocumentLLM,
    module_name: str,
    *,
    use_concurrency: bool = False,
    max_items_per_call: int = 0,
    max_paragraphs_to_analyze_per_call: int = 0,
    benchmark_name: str | None = None,
) -> _BenchmarkScores:
    """
    Runs the benchmark using a document+extraction targets provider module.

    The provider module must export `build_benchmark_document()` returning
    a ready `Document` (with extraction targets added).

    :param llm: The main LLM used for extraction.
    :type llm: DocumentLLM
    :param judge_llm: The LLM used to judge the extraction quality.
    :type judge_llm: DocumentLLM
    :param module_name: The name of the module providing the benchmark document and targets.
    :type module_name: str
    :param use_concurrency: Whether to use concurrency during extraction.
    :type use_concurrency: bool, optional
    :param max_items_per_call: Maximum number of items to extract per call (0 for no limit).
    :type max_items_per_call: int, optional
    :param max_paragraphs_to_analyze_per_call: Maximum number of paragraphs to analyze per call (0 for no limit).
    :type max_paragraphs_to_analyze_per_call: int, optional
    :param benchmark_name: Optional name for the benchmark run.
    :type benchmark_name: str or None, optional
    :return: Aggregated benchmark scores.
    :rtype: _BenchmarkScores
    """

    # Enforce text-only constraint for main LLM
    if llm.is_group:
        raise ValueError(
            "Group roles are not allowed for the main LLM in the benchmark."
        )
    if str(llm.role).endswith("_vision"):
        raise ValueError(
            "Vision role is not allowed for the main LLM in the text-only benchmark."
        )

    # Enforce text-only constraint for judge LLM
    if judge_llm.is_group:
        raise ValueError(
            "Group roles are not allowed for the judge LLM in the benchmark."
        )
    if str(judge_llm.role).endswith("_vision"):
        raise ValueError(
            "Vision role is not allowed for the judge LLM in the text-only benchmark."
        )

    # Unset default system message on the judge LLM (use vanilla LLM for judge)
    judge_llm.system_message = ""

    document = _load_benchmark_document(module_name)

    # Align target roles with the main LLM role
    for a in document.aspects:
        a.llm_role = llm.role
    for c in document.concepts:
        c.llm_role = llm.role

    # Extraction
    try:
        llm.extract_aspects_from_document(
            document=document,
            use_concurrency=use_concurrency,
            max_items_per_call=max_items_per_call,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )
        llm.extract_concepts_from_document(
            document=document,
            use_concurrency=use_concurrency,
            max_items_per_call=max_items_per_call,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )
    except Exception as e:
        raise AssertionError(f"Benchmark failed at extraction stage: {e}") from e

    # Partition by metadata
    simple_aspects = [
        a
        for a in document.aspects
        if a.custom_data.get("benchmark_category") == "simple_aspect"
    ]
    reasoning_aspects = [
        a
        for a in document.aspects
        if a.custom_data.get("benchmark_category") == "reasoning_aspect"
    ]
    hallucination_aspects = [
        a
        for a in document.aspects
        if a.custom_data.get("benchmark_category") == "hallucination_aspect"
    ]

    simple_concepts = [
        c
        for c in document.concepts
        if c.custom_data.get("benchmark_category") == "simple_concept"
    ]
    reasoning_concepts = [
        c
        for c in document.concepts
        if c.custom_data.get("benchmark_category") == "reasoning_concept"
    ]
    hallucination_concepts = [
        c
        for c in document.concepts
        if c.custom_data.get("benchmark_category") == "hallucination_concept"
    ]

    # Presence scoring
    def _presence_score_must_have(objs: list) -> float:
        """
        Calculates the percentage of objects in the list that have extracted items.

        :param objs: List of objects, each expected to have an 'extracted_items' attribute.
        :type objs: list
        :return: Percentage of objects with extracted items.
        :rtype: float
        """
        return _pct(len([o for o in objs if o.extracted_items]), len(objs))

    def _presence_score_must_not_have(objs: list) -> float:
        """
        Calculates the percentage of objects in the list that do NOT have extracted items.

        :param objs: List of objects, each expected to have an 'extracted_items' attribute.
        :type objs: list
        :return: Percentage of objects without extracted items.
        :rtype: float
        """
        return _pct(len([o for o in objs if not o.extracted_items]), len(objs))

    presence_simple_aspects = _presence_score_must_have(simple_aspects)
    presence_reasoning_aspects = _presence_score_must_have(reasoning_aspects)
    presence_simple_concepts = _presence_score_must_have(simple_concepts)
    presence_reasoning_concepts = _presence_score_must_have(reasoning_concepts)
    presence_hallucination_aspects = _presence_score_must_not_have(
        hallucination_aspects
    )
    presence_hallucination_concepts = _presence_score_must_not_have(
        hallucination_concepts
    )

    # Judge scoring for non-hallucination
    ratings_by_target: dict[tuple[str, str], int] = {}
    justifications_by_target: dict[tuple[str, str], str] = {}

    logger.info("== Judge flow started ==")
    for a in simple_aspects + reasoning_aspects:
        rating, justification = _judge_target(
            judge_llm, "aspect", a.name, _items_block(a), document.raw_text
        )
        ratings_by_target[("aspect", a.name)] = rating
        if justification:
            justifications_by_target[("aspect", a.name)] = justification

    for c in simple_concepts + reasoning_concepts:
        rating, justification = _judge_target(
            judge_llm, "concept", c.name, _items_block(c), document.raw_text
        )
        ratings_by_target[("concept", c.name)] = rating
        if justification:
            justifications_by_target[("concept", c.name)] = justification
    logger.info("== Judge flow completed ==")

    def _avg_rating_percent(objs: list) -> float:
        """
        Calculates the average judge rating for a list of objects (aspects or concepts)
        as a percentage.

        :param objs: List of objects (Aspect or _Concept), each expected to have
            a 'name' attribute.
        :type objs: list
        :return: The average rating as a percentage of the ideal judge rating.
        :rtype: float
        """
        if not objs:
            return 0.0
        total = 0
        for o in objs:
            key = ("aspect" if isinstance(o, Aspect) else "concept", o.name)
            total += int(ratings_by_target.get(key, 0))
        return _pct(total, IDEAL_JUDGE_RATING * len(objs))

    judge_simple_aspects = _avg_rating_percent(simple_aspects)
    judge_reasoning_aspects = _avg_rating_percent(reasoning_aspects)
    judge_simple_concepts = _avg_rating_percent(simple_concepts)
    judge_reasoning_concepts = _avg_rating_percent(reasoning_concepts)

    # Non-compliance collections
    non_compliant_presence_aspects_must_have = [
        a.name for a in (simple_aspects + reasoning_aspects) if not a.extracted_items
    ]
    non_compliant_presence_concepts_must_have = [
        c.name for c in (simple_concepts + reasoning_concepts) if not c.extracted_items
    ]
    non_compliant_presence_hallucination_aspects = [
        a.name for a in hallucination_aspects if a.extracted_items
    ]
    non_compliant_presence_hallucination_concepts = [
        c.name for c in hallucination_concepts if c.extracted_items
    ]

    non_compliant_judge_aspects = [
        ("aspect", a.name)
        for a in (simple_aspects + reasoning_aspects)
        if int(ratings_by_target.get(("aspect", a.name), 0)) < IDEAL_JUDGE_RATING
    ]
    non_compliant_judge_concepts = [
        ("concept", c.name)
        for c in (simple_concepts + reasoning_concepts)
        if int(ratings_by_target.get(("concept", c.name), 0)) < IDEAL_JUDGE_RATING
    ]

    # Aggregate
    presence_overall = round(
        (
            presence_simple_aspects
            + presence_reasoning_aspects
            + presence_simple_concepts
            + presence_reasoning_concepts
            + presence_hallucination_aspects
            + presence_hallucination_concepts
        )
        / 6.0,
        2,
    )
    judge_overall = round(
        (
            judge_simple_aspects
            + judge_reasoning_aspects
            + judge_simple_concepts
            + judge_reasoning_concepts
        )
        / 4.0,
        2,
    )
    total_score = round((presence_overall + judge_overall) / 2.0, 2)

    # Print report
    header = benchmark_name or f"Benchmark Results ({module_name})"
    print(header)
    print("Presence-based scoring")
    print(f"- Simple aspects (must-have): {presence_simple_aspects}%")
    print(f"- Reasoning aspects (must-have): {presence_reasoning_aspects}%")
    print(f"- Simple concepts (must-have): {presence_simple_concepts}%")
    print(f"- Reasoning concepts (must-have): {presence_reasoning_concepts}%")
    print(f"- Hallucination aspects (must-not-have): {presence_hallucination_aspects}%")
    print(
        f"- Hallucination concepts (must-not-have): {presence_hallucination_concepts}%"
    )

    print("Judge-based scoring")
    print(f"- Simple aspects: {judge_simple_aspects}%")
    print(f"- Reasoning aspects: {judge_reasoning_aspects}%")
    print(f"- Simple concepts: {judge_simple_concepts}%")
    print(f"- Reasoning concepts: {judge_reasoning_concepts}%")

    print("Total score")
    print(f"{total_score}%")

    print("Presence-based non-compliance")
    print(
        f"- Must-have aspects with no items: {len(non_compliant_presence_aspects_must_have)}"
    )
    if non_compliant_presence_aspects_must_have:
        print("  Names:")
        for name in non_compliant_presence_aspects_must_have:
            print(f"    - {name}")
    print(
        f"- Must-have concepts with no items: {len(non_compliant_presence_concepts_must_have)}"
    )
    if non_compliant_presence_concepts_must_have:
        print("  Names:")
        for name in non_compliant_presence_concepts_must_have:
            print(f"    - {name}")
    print(
        f"- Hallucination aspects with items: {len(non_compliant_presence_hallucination_aspects)}"
    )
    if non_compliant_presence_hallucination_aspects:
        print("  Names:")
        for name in non_compliant_presence_hallucination_aspects:
            print(f"    - {name}")
    print(
        f"- Hallucination concepts with items: {len(non_compliant_presence_hallucination_concepts)}"
    )
    if non_compliant_presence_hallucination_concepts:
        print("  Names:")
        for name in non_compliant_presence_hallucination_concepts:
            print(f"    - {name}")

    print("Judge non-compliance (rating < ideal)")
    print(f"- Aspects below ideal rating: {len(non_compliant_judge_aspects)}")
    for _, name in non_compliant_judge_aspects:
        rating = ratings_by_target.get(("aspect", name), 0)
        just = justifications_by_target.get(("aspect", name), "")
        print(f"    - {name}: rating={rating}")
        if just:
            print(f"      Justification: {just}")
    print(f"- Concepts below ideal rating: {len(non_compliant_judge_concepts)}")
    for _, name in non_compliant_judge_concepts:
        rating = ratings_by_target.get(("concept", name), 0)
        just = justifications_by_target.get(("concept", name), "")
        print(f"    - {name}: rating={rating}")
        if just:
            print(f"      Justification: {just}")

    # Threshold enforcement
    if total_score < MIN_SCORE_THRESHOLD:
        raise AssertionError(
            f"Benchmark failed: total score {total_score}% is below {MIN_SCORE_THRESHOLD}% threshold"
        )

    return _BenchmarkScores(
        presence_simple_aspects=presence_simple_aspects,
        presence_reasoning_aspects=presence_reasoning_aspects,
        presence_simple_concepts=presence_simple_concepts,
        presence_reasoning_concepts=presence_reasoning_concepts,
        presence_hallucination_aspects=presence_hallucination_aspects,
        presence_hallucination_concepts=presence_hallucination_concepts,
        judge_simple_aspects=judge_simple_aspects,
        judge_reasoning_aspects=judge_reasoning_aspects,
        judge_simple_concepts=judge_simple_concepts,
        judge_reasoning_concepts=judge_reasoning_concepts,
        presence_non_compliant_aspects_must_have_count=len(
            non_compliant_presence_aspects_must_have
        ),
        presence_non_compliant_concepts_must_have_count=len(
            non_compliant_presence_concepts_must_have
        ),
        presence_non_compliant_hallucination_aspects_count=len(
            non_compliant_presence_hallucination_aspects
        ),
        presence_non_compliant_hallucination_concepts_count=len(
            non_compliant_presence_hallucination_concepts
        ),
        judge_non_compliant_aspects_count=len(non_compliant_judge_aspects),
        judge_non_compliant_concepts_count=len(non_compliant_judge_concepts),
        total=total_score,
    )
