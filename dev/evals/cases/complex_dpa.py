#
# ContextGem
#
# Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Extraction schema for the complex DPA evaluation case."""

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


def build_complex_dpa_pipeline() -> ExtractionPipeline:
    """Builds a fresh extraction template without source text or gold annotations.

    :return: The reusable contract extraction pipeline.
    :rtype: ExtractionPipeline
    """
    return ExtractionPipeline(
        aspects=[
            Aspect(
                name="Amendment effectiveness and precedence",
                description=(
                    "Provisions governing when the amendments take effect and how"
                    " conflicts between the amendments, their annexes, and the "
                    "DPA are resolved"
                ),
            ),
            Aspect(
                name="Personal data breach notification",
                description=(
                    "Clauses governing personal data breach notification, "
                    "including any amendments to them"
                ),
            ),
            Aspect(
                name="Incident IR-27",
                description=(
                    "All governing notification rules, effective-date and clock "
                    "conventions, and operational records needed to assess IR-27,"
                    " including the original and amended notification obligations"
                ),
                concepts=[
                    LabelConcept(
                        name="Applicable initial notification regime",
                        description=(
                            "The initial-notification regime applicable to IR-27"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "standard",
                            "expedited",
                        ],
                    ),
                    NumericalConcept(
                        name="Initial notice lateness in minutes",
                        description=(
                            "Minutes by which IR-27's first notice exceeded its "
                            "applicable contractual outer time limit; zero if it "
                            "met that limit. Assess the outer limit only."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    DateConcept(
                        name="Completed follow-up report deadline",
                        description=(
                            "The calendar date on which IR-27's completed "
                            "follow-up report is due"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Retention TR-27",
                description=(
                    "All deletion and retention rules, effective-date and clock "
                    "conventions, and operational records needed to resolve "
                    "TR-27, including the original and amended deletion and "
                    "backup terms"
                ),
                concepts=[
                    DateConcept(
                        name="Held ORD records deletion deadline",
                        description=(
                            "The latest deletion date for ORD records 001 through"
                            " 050 in TR-27"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    DateConcept(
                        name="BK-0630 deletion deadline",
                        description=(
                            "The latest deletion date for backup BK-0630 in TR-27"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    LabelConcept(
                        name=("Permitted uses of held ORD records on 1 September 2027"),
                        description=(
                            "Every permitted use of ORD records 001 through 050 "
                            "on 1 September 2027 in TR-27"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "service delivery",
                            "legal preservation",
                            "analytics",
                        ],
                        classification_type="multi_label",
                    ),
                ],
            ),
            Aspect(
                name="Supplier approvals in May 2027",
                description=(
                    "All authorisation and location rules, effective-date "
                    "provisions, supplier lists and change records needed to "
                    "establish approved services and actual data locations in May"
                    " 2027, including earlier approvals, withdrawals and "
                    "emergency permissions"
                ),
                concepts=[
                    DateConcept(
                        name="Nordlicht infrastructure approval start",
                        description=(
                            "The first date Nordlicht Cloud AB is authorised to "
                            "provide the Stockholm infrastructure service"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    StringConcept(
                        name="Ordinary customer email providers on 10 May 2027",
                        description=(
                            "The legal names of all Sub-processors authorised for"
                            " ordinary customer transactional email at 12:00 UTC "
                            "on 10 May 2027, one item per legal entity"
                        ),
                        add_justifications=True,
                        add_references=True,
                    ),
                    LabelConcept(
                        name="Authorised data locations on 15 May 2027",
                        description=(
                            "All countries with both actual Personal Data "
                            "processing and effective authorisation on 15 May "
                            "2027"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "Norway",
                            "Estonia",
                            "Lithuania",
                            "France",
                            "Denmark",
                            "Sweden",
                            "Finland",
                            "Germany",
                            "Ireland",
                        ],
                        classification_type="multi_label",
                    ),
                    DateConcept(
                        name=("Written consent date for Nordlicht email in Finland"),
                        description=(
                            "The date the Controller gave written consent for "
                            "Nordlicht Cloud AB's email service in Finland, if "
                            "such consent is documented through 15 May 2027"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Liability LC-27",
                description=(
                    "All applicable liability rules, effective-date provisions, "
                    "fee and payment records, and established new claims needed "
                    "to calculate LC-27, including earlier cap terms and "
                    "carve-outs"
                ),
                concepts=[
                    NumericalConcept(
                        name="Security cap before earlier payments in EUR",
                        description=(
                            "The security liability cap applicable to LC-27 "
                            "before deducting earlier claim payments, in EUR"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    NumericalConcept(
                        name="Ordinary cap remaining before new claims in EUR",
                        description=(
                            "The ordinary liability pool remaining for LC-27 "
                            "before paying claims C-1 through C-5, in EUR"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    NumericalConcept(
                        name="Total payable on new claims in EUR",
                        description=(
                            "The total amount payable on LC-27 claims C-1 through"
                            " C-5 after applying all contractual allocation "
                            "rules, in EUR"
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                ],
            ),
        ],
        concepts=[
            NumericalConcept(
                name="Breach notification deadline in force on 15 September 2026",
                description=(
                    "The maximum number of hours within which the Processor must "
                    "notify the Controller of a personal data breach, under the "
                    "provisions in force on 15 September 2026"
                ),
                add_justifications=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            NumericalConcept(
                name="Breach notification deadline in force on 1 November 2026",
                description=(
                    "The maximum number of hours within which the Processor must "
                    "notify the Controller of a personal data breach, under the "
                    "provisions in force on 1 November 2026"
                ),
                add_justifications=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            NumericalConcept(
                name="Audit notice period in force on 1 November 2026",
                description=(
                    "The minimum number of days of prior written notice required "
                    "for a routine on-site audit, under the provisions in force "
                    "on 1 November 2026"
                ),
                add_justifications=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            NumericalConcept(
                name="Liability cap in force on 1 November 2026",
                description=(
                    "The Processor's aggregate liability cap in EUR for breaches "
                    "of the agreement, under the provisions in force on 1 "
                    "November 2026, assuming the Controller paid EUR 700,000 in "
                    "fees under the MSA during the 12 months preceding the claim"
                ),
                add_justifications=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            BooleanConcept(
                name="Liability cap covers confidentiality breaches",
                description=(
                    "Whether, under the provisions in force on 1 November 2026, "
                    "the Processor's aggregate liability cap applies to breaches "
                    "of the confidentiality obligations"
                ),
                add_justifications=True,
                singular_occurrence=True,
            ),
            BooleanConcept(
                name="Meridian Mail approved on 1 November 2026",
                description=(
                    "Whether Meridian Mail Ltd is an approved Sub-processor on 1 "
                    "November 2026"
                ),
                add_justifications=True,
                singular_occurrence=True,
            ),
            BooleanConcept(
                name="Meridian Mail approved on 1 February 2027",
                description=(
                    "Whether Meridian Mail Ltd is an approved Sub-processor on 1 "
                    "February 2027"
                ),
                add_justifications=True,
                singular_occurrence=True,
            ),
            NumericalConcept(
                name="Distinct data locations count as of 1 February 2027",
                description=(
                    "The total number of distinct countries in which Personal "
                    "Data is stored or processed under the agreement as of 1 "
                    "February 2027 (including its annexes and amendments)"
                ),
                add_justifications=True,
                singular_occurrence=True,
                numeric_type="int",
            ),
            DateConcept(
                name="Deletion deadline for year-end termination",
                description=(
                    "The latest calendar date by which Personal Data must be "
                    "deleted or returned if the MSA terminates on 31 December "
                    "2026, applying the provisions in force on that date"
                ),
                add_justifications=True,
                singular_occurrence=True,
            ),
            JsonObjectConcept(
                name="Approved sub-processors as of 1 February 2027",
                description=(
                    "The Sub-processors approved as of 1 February 2027, their "
                    "countries of registration, and a concise description of the "
                    "service each provides"
                ),
                structure={"name": str, "country": str, "service": str},
            ),
            StringConcept(
                name="Breach notification obligation summary",
                description=(
                    "A concise summary (1-2 sentences, in your own words) of the "
                    "Processor's personal data breach notification obligation as "
                    "in force on 1 November 2026"
                ),
                add_justifications=True,
                singular_occurrence=True,
            ),
        ],
    )
