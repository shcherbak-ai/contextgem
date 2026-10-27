# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Extraction schema for the complex multilingual contract evaluation case."""

from __future__ import annotations

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


def build_complex_multilingual_pipeline() -> ExtractionPipeline:
    """Builds a fresh extraction template without source text or gold annotations.

    The document is a distribution agreement in German with an annex in English
    and an amendment in French, so that every output language other than these
    needs translation.

    :return: The reusable distribution agreement extraction pipeline.
    :rtype: ExtractionPipeline
    """
    return ExtractionPipeline(
        aspects=[
            Aspect(
                name="Territory and minimum purchases",
                description=(
                    "The Distributor's territory, its exclusivity and the limits on "
                    "its sales outside the territory, the minimum purchase "
                    "quantities and the consequences of a shortfall, with their "
                    "changes by the amendment"
                ),
                add_justifications=True,
                concepts=[
                    LabelConcept(
                        name="Countries of the exclusive territory in 2028",
                        description=(
                            "The countries in which the Distributor has exclusivity "
                            "throughout 2028 under the agreement as amended. Leave "
                            "out a country in which it may only sell "
                            "non-exclusively."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "France",
                            "Monaco",
                            "Belgium",
                            "Luxembourg",
                            "Switzerland",
                            "Germany",
                        ],
                        classification_type="multi_label",
                    ),
                    NumericalConcept(
                        name="Minimum purchase quantity for 2028 in pumps",
                        description=(
                            "The minimum number of pumps that the Distributor must "
                            "purchase in the calendar year 2028 under the agreement "
                            "as amended"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    StringConcept(
                        name="Consequence of a minimum purchase shortfall",
                        description=(
                            "What the Manufacturer may do if the Distributor "
                            "purchases less than the minimum quantity of a calendar "
                            "year, and what it may not, in one sentence"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Delivery and transport risk",
                description=(
                    "The delivery terms and the place of delivery, the lead times, "
                    "the transport costs, and the passing of the risk of the goods "
                    "to the Distributor, with their changes by the amendment"
                ),
                add_justifications=True,
                concepts=[
                    LabelConcept(
                        name="Incoterm of a delivery on 12 April 2028",
                        description=(
                            "The Incoterms 2020 rule that applies to a delivery of "
                            "Contract Products made on 12 April 2028"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=["EXW", "FCA", "CPT", "DAP", "DDP"],
                        classification_type="multi_class",
                    ),
                    StringConcept(
                        name="Passing of the transport risk",
                        description=(
                            "Who bears the risk of loss of or damage to the goods "
                            "in transit, and when it passes to the Distributor, for "
                            "deliveries before and after the change of the delivery "
                            "terms, in one or two sentences"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Prices and payment",
                description=(
                    "The list prices of the Contract Products, the discounts, the "
                    "initial stocking order, the invoicing and the payment terms, "
                    "and the rules that decide between the German agreement and its "
                    "Annex 1 where they differ"
                ),
                add_justifications=True,
                concepts=[
                    NumericalConcept(
                        name="Net value of the initial stocking order in EUR",
                        description=(
                            "The amount of the initial stocking order in EUR at "
                            "the list prices that apply, after the introductory "
                            "discount, excluding VAT and any early-payment discount"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    DateConcept(
                        name="Payment due date of the initial stocking order invoice",
                        description=(
                            "The last day of the payment term of the invoice for "
                            "the initial stocking order, i.e. the date by which it "
                            "must be paid in full without deduction"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Warranty and product performance",
                description=(
                    "The inspection of the goods and the notice of defects, the "
                    "warranty periods and the Wear Parts, the guaranteed technical "
                    "data of the Contract Products, the remedies for defects, the "
                    "supply of spare parts, and the rules that decide between the "
                    "German agreement and its Annex 1 where they differ"
                ),
                add_justifications=True,
                concepts=[
                    NumericalConcept(
                        name="Guaranteed minimum efficiency of HP-500 at BEP in %",
                        description=(
                            "The minimum efficiency at the best efficiency point, "
                            "in percent, that the Manufacturer guarantees for HP-500 "
                            "pumps, as it applies under the agreement"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    NumericalConcept(
                        name="Warranty period of a mechanical seal in months",
                        description=(
                            "The warranty period, in months from the passing of "
                            "risk, of a mechanical seal of a Contract Product"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    StringConcept(
                        name="Spare parts supply period",
                        description=(
                            "How long the Manufacturer must supply spare parts for "
                            "the Contract Products, in one sentence"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
        ],
        concepts=[
            DateConcept(
                name="Amendment No. 1 effective date",
                description="The date on which Amendment No. 1 enters into force",
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            StringConcept(
                name="Changes made by Amendment No. 1",
                description=(
                    "What Amendment No. 1 changes in the agreement, in one sentence"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            StringConcept(
                name="Governing law and jurisdiction",
                description=(
                    "The law that governs the agreement and the courts that have "
                    "jurisdiction over its disputes"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            StringConcept(
                name="Prevailing text between the agreement and Annex 1",
                description=(
                    "Which text prevails where the German agreement and its English "
                    "Annex 1 differ"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            BooleanConcept(
                name="HP-700 is a Contract Product",
                description=(
                    "Whether the pumps of the HP-700 series are Contract Products "
                    "under the agreement as amended"
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            JsonObjectConcept(
                name="Contract Products and their list prices",
                description=(
                    "One object per pump series that is a Contract Product: its "
                    "series designation (e.g. HP-200), its description as Annex 1 "
                    "gives it, and its list price in EUR net per unit as it applies "
                    "under the agreement"
                ),
                add_justifications=True,
                add_references=True,
                structure={"series": str, "description": str, "list_price_eur": int},
            ),
        ],
    )
