# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Extraction schema for the complex microcontroller documentation evaluation case."""

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
    RatingConcept,
    StringConcept,
)


# Labels of the errata of the device
_ERRATA_IDS = [f"ES-0{number}" for number in range(1, 10)]


def build_complex_mcu_pipeline() -> ExtractionPipeline:
    """Builds a fresh extraction template without source text or gold annotations.

    :return: The reusable technical documentation extraction pipeline.
    :rtype: ExtractionPipeline
    """
    return ExtractionPipeline(
        aspects=[
            Aspect(
                name="Device limitations of revision C",
                description=(
                    "Tessaline's documentation of the device limitations that are "
                    "present in silicon revision C: their conditions, workarounds "
                    "and impact classes, the definition of the impact classes, and "
                    "the changes of their status or impact class between documents"
                ),
                concepts=[
                    JsonObjectConcept(
                        name="Limitations present in revision C",
                        description=(
                            "One object per device limitation present in silicon "
                            "revision C: its ID (e.g. ES-01), the function that it "
                            "affects as the errata sheet names it, its current "
                            "impact class, and the silicon revisions in which it is "
                            "present"
                        ),
                        add_references=True,
                        structure={
                            "id": str,
                            "function": str,
                            "impact_class": int,
                            "affected_revisions": list[Literal["A", "B", "C"]],
                        },
                    ),
                    RatingConcept(
                        name="Impact class of ES-08",
                        description=(
                            "The current impact class of the device limitation "
                            "ES-08, on the scale of the errata sheet"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        rating_scale=(1, 5),
                    ),
                    BooleanConcept(
                        name="ES-09 present in revision C",
                        description=(
                            "Whether the device limitation ES-09 is present in "
                            "silicon revision C"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    StringConcept(
                        name="ES-08 workaround",
                        description="The workaround of ES-08, in one sentence",
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Flash memory specifications",
                description=(
                    "Tessaline's specifications of the embedded flash memory: read "
                    "latency and wait states, endurance and data retention, with "
                    "the corrections and changes that apply to them"
                ),
                concepts=[
                    NumericalConcept(
                        name="HCLK maximum for 2 wait states in VOS1 in MHz",
                        description=(
                            "The highest HCLK frequency, in MHz, at which 2 flash "
                            "wait states are sufficient in the voltage-scaling range "
                            "VOS1"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    NumericalConcept(
                        name="Revision C retention after 10 000 cycles at 85 °C",
                        description=(
                            "The guaranteed minimum flash data retention, in years, "
                            "of a silicon revision C device after 10 000 cycles at "
                            "Ta = 85 °C"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    NumericalConcept(
                        name="Revision C retention after 1 000 cycles at 105 °C",
                        description=(
                            "The guaranteed minimum flash data retention, in years, "
                            "of a silicon revision C device after 1 000 cycles at "
                            "Ta = 105 °C"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                ],
            ),
            Aspect(
                name="DR-114 clock configuration",
                description=(
                    "What is needed to determine the highest SYSCLK frequency "
                    "permitted for the DR-114 production boards, whether their PLL1 "
                    "configuration is within the specification, and how many flash "
                    "wait states they require: the clock, PLL, voltage-scaling, "
                    "bus-clock and flash-latency rules, the errata, corrections and "
                    "changes that affect them, and the DR-114 records"
                ),
                concepts=[
                    NumericalConcept(
                        name="DR-114 maximum permitted SYSCLK in MHz",
                        description=(
                            "The highest SYSCLK frequency, in MHz, that the "
                            "documentation permits for the DR-114 production boards "
                            "in any voltage-scaling range, under their supply and "
                            "ambient conditions"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    BooleanConcept(
                        name="DR-114 PLL1 configuration within specification",
                        description=(
                            "Whether every PLL1 setting recorded for the DR-114 "
                            "boards is within the specification"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name="DR-114 required flash wait states",
                        description=(
                            "The smallest number of flash wait states (LATENCY) that "
                            "the documentation permits for the DR-114 production "
                            "boards with their configured clocks"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                ],
            ),
            Aspect(
                name="DR-114 thermal assessment",
                description=(
                    "What is needed to compute the junction temperature of the "
                    "DR-114 production boards and to check it against Halyard's "
                    "design rule HDR-7: the junction-temperature method, "
                    "the thermal-resistance, current-consumption and "
                    "temperature-limit values, and the DR-114 records"
                ),
                concepts=[
                    NumericalConcept(
                        name="DR-114 junction temperature in °C",
                        description=(
                            "The junction temperature of the DR-114 production "
                            "boards at their highest ambient temperature, computed "
                            "with the method of the datasheet, in °C rounded to the "
                            "nearest whole degree"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    BooleanConcept(
                        name="DR-114 meets HDR-7",
                        description=(
                            "Whether the DR-114 production boards meet Halyard's "
                            "design rule HDR-7"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="DR-114 errata and firmware",
                description=(
                    "What is needed to decide which errata apply to the DR-114 "
                    "production boards, the maximum INL of their ADC configuration, "
                    "and whether firmware FW 2.3.0 meets Halyard's design rule "
                    "HDR-9: the device identification, the errata and their "
                    "conditions, the ADC sampling and accuracy rules, and the DR-114 "
                    "records"
                ),
                concepts=[
                    LabelConcept(
                        name="Errata applicable to the DR-114 production boards",
                        description=(
                            "Every erratum that is present in the silicon revision "
                            "of the DR-114 production boards and whose conditions "
                            "their configuration meets"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=_ERRATA_IDS,
                        classification_type="multi_label",
                    ),
                    NumericalConcept(
                        name="DR-114 maximum ADC INL in LSB",
                        description=(
                            "The maximum magnitude of the INL, in LSB, that the "
                            "documentation gives for the ADC configuration of "
                            "the DR-114 production boards with firmware FW 2.3.0"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    BooleanConcept(
                        name="FW 2.3.0 meets HDR-9 for the DR-114 production boards",
                        description=(
                            "Whether firmware FW 2.3.0 meets Halyard's design rule "
                            "HDR-9 for the DR-114 production boards"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="DR-115 errata and PLL1 limits",
                description=(
                    "What is needed to decide which errata apply to the DR-115 "
                    "boards, the highest PLL1 VCO frequency permitted for them, and "
                    "whether their PLL1 configuration is within all the limits that "
                    "apply to them: the device identification, the errata and their "
                    "conditions, the PLL rules, and the DR-115 records"
                ),
                concepts=[
                    LabelConcept(
                        name="Errata applicable to the DR-115 boards",
                        description=(
                            "Every erratum that is present in the silicon revision "
                            "of the DR-115 boards and whose conditions their "
                            "configuration and environment meet"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=_ERRATA_IDS,
                        classification_type="multi_label",
                    ),
                    NumericalConcept(
                        name="DR-115 maximum PLL1 VCO frequency in MHz",
                        description=(
                            "The highest PLL1 VCO frequency, in MHz, permitted for "
                            "the DR-115 boards, including the limits of the errata "
                            "that apply to them"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    BooleanConcept(
                        name="DR-115 PLL1 configuration within its limits",
                        description=(
                            "Whether the PLL1 configuration recorded for the DR-115 "
                            "boards is within every limit that applies to them, "
                            "including the workarounds of the errata"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Revision B end of life",
                description=(
                    "What is needed to establish Halyard's last-time-buy date and "
                    "the last-shipment date of revision B devices, and whether "
                    "purchase order PO-5521 is a valid last-time-buy order: the PCN "
                    "terms and their amendment, the counting of calendar days, and "
                    "the Halyard purchasing records"
                ),
                concepts=[
                    DateConcept(
                        name="Halyard last-time-buy date for revision B",
                        description=(
                            "The last date on which Halyard Drives AS can place "
                            "a last-time-buy order for revision B devices"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    DateConcept(
                        name="Last-shipment date of revision B",
                        description=(
                            "The last date on which revision B devices are shipped "
                            "to Halyard Drives AS"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    BooleanConcept(
                        name="PO-5521 valid last-time-buy order",
                        description=(
                            "Whether purchase order PO-5521 is accepted as "
                            "a last-time-buy order for revision B devices"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
        ],
        concepts=[
            NumericalConcept(
                name="Lowest HBM rating of a QFN-48 pin in V",
                description=(
                    "The lowest human-body-model ESD rating, in V, of any pin of "
                    "a TSV7H device in the QFN-48 package"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            NumericalConcept(
                name="Revision C BOR level 2 minimum falling threshold in V",
                description=(
                    "The minimum falling threshold of BOR level 2, in V, of silicon "
                    "revision C devices"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
                numeric_type="float",
            ),
            BooleanConcept(
                name="VOS0 permitted at a VDD of 1.8 V",
                description=(
                    "Whether the voltage-scaling range VOS0 can be used with a VDD "
                    "of 1.8 V"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            LabelConcept(
                name="Marking code of revision C",
                description=(
                    "The revision code on the top marking of the production devices "
                    "of silicon revision C"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
                labels=["A", "B", "C", "X", "Y"],
                classification_type="multi_class",
            ),
            NumericalConcept(
                name="DR-115 junction temperature in °C",
                description=(
                    "The junction temperature of the DR-115 boards at their highest "
                    "ambient temperature, computed with the method of the datasheet, "
                    "in °C rounded to the nearest whole degree"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            DateConcept(
                name="Start of the first revision C date code",
                description=(
                    "The first day, a Monday, of the week given by the first date "
                    "code from which revision C devices are shipped"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
        ],
    )
