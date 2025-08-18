"""
Dev contract benchmark set: document + interconnected aspects/concepts.
"""

from __future__ import annotations

from pathlib import Path

from contextgem import (
    Aspect,
    Document,
    JsonObjectConcept,
    LabelConcept,
    StringConcept,
    StringExample,
)


_TEST_DOC_PATH = Path(__file__).parent / "dev_contract.txt"


def build_benchmark_document() -> Document:
    """
    Builds and returns the benchmark Document for the dev contract benchmark set.

    :return: A Document instance containing the raw contract text and pre-defined
        aspects/concepts for benchmarking.
    :rtype: Document
    """

    with open(_TEST_DOC_PATH, encoding="utf-8") as f:
        raw_text = f.read().strip()

    doc = Document(raw_text=raw_text)

    # Simple aspects (must be extracted)
    simple_aspects = [
        Aspect(
            name="Limitation of Liability",
            description=(
                "Clauses that limit or cap liability, disclaim consequential damages, "
                "or otherwise restrict remedies."
            ),
            custom_data={"benchmark_category": "simple_aspect"},
        ),
        Aspect(
            name="Payment Terms",
            description=(
                "Clauses that define fees, invoicing cadence, payment due dates, and late charges."
            ),
            custom_data={"benchmark_category": "simple_aspect"},
        ),
        Aspect(
            name="Confidentiality and Non-Disclosure",
            description=(
                "Clauses describing confidentiality obligations and restrictions on disclosure."
            ),
            custom_data={"benchmark_category": "simple_aspect"},
        ),
    ]

    # Reasoning aspects (must be extracted)
    reasoning_aspects = [
        Aspect(
            name="Governing Law Inconsistencies",
            description=("Contradictions or overrides in governing law provisions."),
            reference_depth="paragraphs",
            add_justifications=True,
            custom_data={"benchmark_category": "reasoning_aspect"},
        ),
        Aspect(
            name="Open Source Copyleft Exposure",
            description="OSS provisions that may introduce copyleft risk or unusual exceptions.",
            reference_depth="paragraphs",
            add_justifications=True,
            custom_data={"benchmark_category": "reasoning_aspect"},
        ),
        Aspect(
            name="Aggressive Deemed Acceptance",
            description="Unusually short review/acceptance windows and deemed-acceptance constructs.",
            reference_depth="paragraphs",
            add_justifications=True,
            custom_data={"benchmark_category": "reasoning_aspect"},
        ),
    ]

    # Simple concepts (must be extracted)
    simple_concepts = [
        StringConcept(
            name="Party Details",
            description="Party name and role.",
            examples=[StringExample(content="Company A (Client)")],
            custom_data={"benchmark_category": "simple_concept"},
        ),
        JsonObjectConcept(
            name="Key Service Levels",
            description="Key service level targets.",
            structure={
                "availability_target": float,
                "severity_levels": [{"level": int, "response_hours": float}],
            },
            singular_occurrence=True,
            custom_data={"benchmark_category": "simple_concept"},
        ),
        LabelConcept(
            name="Primary Agreement Type",
            description="Primary type of agreement.",
            labels=[
                "web application development",
                "software license",
                "consulting services",
                "data processing agreement",
                "other",
            ],
            classification_type="multi_class",
            singular_occurrence=True,
            custom_data={"benchmark_category": "simple_concept"},
        ),
    ]

    # Reasoning concepts (must be extracted)
    reasoning_concepts = [
        StringConcept(
            name="Anomalous Provisions",
            description="Provisions that appear as anomalies compared to common practice.",
            add_references=True,
            reference_depth="paragraphs",
            add_justifications=True,
            custom_data={"benchmark_category": "reasoning_concept"},
        ),
        JsonObjectConcept(
            name="Risk Assessment Summary (Developer)",
            description=(
                "Potential risks with key details from a developer's perspective. "
                "Provision reference should be a reference to the provision that entails the risk "
                "(e.g. Section A, Section 1.1, etc.). Risk description should be a short description "
                "of the risk. Risk severity should be a number between 1 and 5, where 1 is the "
                "lowest severity and 5 is the highest severity."
            ),
            structure={
                "provision_reference": str,
                "risk_description": str,
                "risk_severity": int,
            },
            add_references=True,
            reference_depth="paragraphs",
            add_justifications=True,
            custom_data={"benchmark_category": "reasoning_concept"},
        ),
        LabelConcept(
            name="Agreement Risk Category (Client)",
            description="Overall agreement risk category, from a client's perspective.",
            labels=["low", "medium", "high"],
            classification_type="multi_class",
            singular_occurrence=True,
            add_references=True,
            reference_depth="paragraphs",
            add_justifications=True,
            custom_data={"benchmark_category": "reasoning_concept"},
        ),
    ]

    # Hallucination aspects (must never be extracted)
    hallucination_aspects = [
        Aspect(
            name="Nuclear Regulatory Compliance",
            description="Clauses specific to nuclear facilities or NRC compliance.",
            custom_data={"benchmark_category": "hallucination_aspect"},
        ),
        Aspect(
            name="Aviation Flight Safety Standards",
            description="Clauses about FAA flight safety or aircraft operation standards.",
            custom_data={"benchmark_category": "hallucination_aspect"},
        ),
        Aspect(
            name="Maritime Salvage Rights",
            description="Clauses on maritime salvage, shipwrecks, or admiralty salvage rights.",
            custom_data={"benchmark_category": "hallucination_aspect"},
        ),
    ]

    # Hallucination concepts (must never be extracted)
    hallucination_concepts = [
        StringConcept(
            name="International Space Treaty Reference",
            description="References to OST, Moon Agreement, or space treaties.",
            custom_data={"benchmark_category": "hallucination_concept"},
        ),
        StringConcept(
            name="Radioactive Waste Disposal Site",
            description="Named locations or licenses for radioactive waste disposal.",
            custom_data={"benchmark_category": "hallucination_concept"},
        ),
        LabelConcept(
            name="Vessel Type Classification",
            description="Classification of maritime vessel types mentioned.",
            labels=["tanker", "bulk carrier", "container ship", "fishing"],
            classification_type="multi_label",  # do not use multi_class as a label will always be returned
            singular_occurrence=True,
            custom_data={"benchmark_category": "hallucination_concept"},
        ),
    ]

    doc.add_aspects(simple_aspects + reasoning_aspects + hallucination_aspects)
    doc.add_concepts(simple_concepts + reasoning_concepts + hallucination_concepts)
    return doc
