# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Extraction schema for the complex stability report evaluation case."""

from __future__ import annotations

from typing import Literal

from contextgem import (
    Aspect,
    BooleanConcept,
    DateConcept,
    ExtractionPipeline,
    JsonObjectConcept,
    LabelConcept,
    NumericalConcept,
    StringConcept,
)


# The batches of the study
_BATCHES = ["V20-2601", "V20-2604", "V20-2607"]


def build_complex_stability_pipeline() -> ExtractionPipeline:
    """Builds a fresh extraction template without source text or gold annotations.

    Most of the answers are in tables whose rows look alike: two specifications
    with the same tests, and a table of results for each batch and condition with
    the same tests and the same number of columns. The concepts of the aspects are
    extracted from the paragraphs of their aspect only, where a table row may come
    without the caption and the header of its table.

    :return: The reusable stability report extraction pipeline.
    :rtype: ExtractionPipeline
    """
    return ExtractionPipeline(
        aspects=[
            Aspect(
                name="Shelf-life acceptance criteria",
                description=(
                    "Identify the acceptance criteria that the stability samples "
                    "are tested against, and the rule on results that do not meet "
                    "them. Leave out the release specification."
                ),
                concepts=[
                    NumericalConcept(
                        name="Shelf-life limit of impurity D in %",
                        description=(
                            "The highest content of impurity D, in %, that "
                            "the shelf-life specification accepts"
                        ),
                        numeric_type="float",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name="Shelf-life limit of water content in %",
                        description=(
                            "The highest water content, in %, that the shelf-life "
                            "specification accepts"
                        ),
                        numeric_type="float",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name="Lower shelf-life limit of assay in %",
                        description=(
                            "The lowest assay, in % of the label claim, that "
                            "the shelf-life specification accepts"
                        ),
                        numeric_type="float",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Stability results of V20-2604",
                description=(
                    "Identify the results of batch V20-2604, in the PVC/Alu "
                    "blister, at each storage condition, and the conclusions that "
                    "the report draws from them in sections 4 and 5."
                ),
                concepts=[
                    NumericalConcept(
                        name=(
                            "Total impurities of V20-2604 after 6 months at "
                            "the accelerated condition in %"
                        ),
                        description=(
                            "The total impurities, in %, of batch V20-2604 after "
                            "6 months at 40 °C/75 % RH"
                        ),
                        numeric_type="float",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name=(
                            "Water content of V20-2604 after 12 months at "
                            "the long-term condition in %"
                        ),
                        description=(
                            "The water content, in %, of batch V20-2604 after "
                            "12 months at 25 °C/60 % RH"
                        ),
                        numeric_type="float",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name=(
                            "Assay of V20-2604 after 12 months at the intermediate "
                            "condition in %"
                        ),
                        description=(
                            "The assay, in % of the label claim, of batch V20-2604 "
                            "after 12 months at 30 °C/65 % RH"
                        ),
                        numeric_type="float",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Results outside the shelf-life specification",
                description=(
                    "Identify the results of the study that do not meet the "
                    "shelf-life specification, as the tables give them, "
                    "the acceptance criteria that they fail, and the investigation "
                    "of each in the report."
                ),
                concepts=[
                    LabelConcept(
                        name="Tests with a result outside the shelf-life specification",
                        description=(
                            "Select every test with at least one result of any "
                            "batch at any condition that does not meet "
                            "the shelf-life specification"
                        ),
                        labels=[
                            "Assay",
                            "Impurity D",
                            "Total impurities",
                            "Dissolution",
                            "Water content",
                        ],
                        classification_type="multi_label",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name="Number of results outside the shelf-life specification",
                        description=(
                            "How many results of the study, counting each test of "
                            "each batch at each condition and time point once, do "
                            "not meet the shelf-life specification"
                        ),
                        numeric_type="int",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    BooleanConcept(
                        name=(
                            "Every result outside the shelf-life specification is "
                            "investigated in section 5"
                        ),
                        description=(
                            "Whether section 5 reports an investigation of each "
                            "result in the tables that does not meet the shelf-life "
                            "specification"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    StringConcept(
                        name=(
                            "Root cause of the results of V20-2604 outside "
                            "the shelf-life specification"
                        ),
                        description=(
                            "The root cause that the investigation found for "
                            "the results of batch V20-2604 that do not meet "
                            "the shelf-life specification, in one sentence"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Shelf-life rules and proposal",
                description=(
                    "Identify the rules of the report for setting the shelf life, "
                    "and the shelf life and the storage statement that it proposes "
                    "for each container closure system."
                ),
                concepts=[
                    NumericalConcept(
                        name="Proposed shelf life of the PVC/Alu blister in months",
                        description=(
                            "The shelf life, in months, that the report proposes "
                            "for the PVC/Alu blister"
                        ),
                        numeric_type="int",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    StringConcept(
                        name="Storage statement of the PVC/Alu blister",
                        description=(
                            "The storage statement that the report proposes for "
                            "the PVC/Alu blister, as worded in the report"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    LabelConcept(
                        name="Rule that gives the proposed shelf life of the PVC/Alu blister",
                        description=(
                            "Select the rule of section 6 under which the proposed "
                            "shelf life of the PVC/Alu blister is set"
                        ),
                        labels=["6.1", "6.2", "6.3"],
                        classification_type="multi_class",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
        ],
        concepts=[
            JsonObjectConcept(
                name="Batches on stability",
                description=(
                    "One object per batch placed on stability: its batch number "
                    "(e.g. V20-2601), its scale, and its container closure system"
                ),
                structure={
                    "id": str,
                    "scale": Literal["pilot", "production"],
                    "container_closure_system": Literal[
                        "HDPE bottle with desiccant",
                        "PVC/Alu blister",
                        "Alu/Alu blister",
                    ],
                },
                add_references=True,
                custom_data={"object_key": "id"},
            ),
            LabelConcept(
                name="Batches with a significant change at the accelerated condition",
                description=(
                    "Select every batch whose results at 40 °C/75 % RH show "
                    "a significant change as section 3.3 defines it, from "
                    "the results in the tables, whatever the narrative of the report "
                    "says"
                ),
                labels=_BATCHES,
                classification_type="multi_label",
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            NumericalConcept(
                name=(
                    "Shelf life supported for the Alu/Alu blister by the rules of "
                    "section 6 in months"
                ),
                description=(
                    "The longest shelf life, in months, that the rules of section 6 "
                    "allow for the Alu/Alu blister, applied to the results in "
                    "the tables, whatever the report proposes"
                ),
                numeric_type="int",
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            BooleanConcept(
                name=(
                    "The conclusion on V20-2607 at the accelerated condition agrees "
                    "with its results"
                ),
                description=(
                    "Whether what the report concludes about batch V20-2607 at "
                    "40 °C/75 % RH agrees with its results in the tables"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            DateConcept(
                name="Date on which V20-2604 was placed at the intermediate condition",
                description=(
                    "The date on which the samples of batch V20-2604 were placed at "
                    "30 °C/65 % RH"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
        ],
    )
