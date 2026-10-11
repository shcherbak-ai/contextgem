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

"""Extraction schema for the complex NDA evaluation case."""

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


def build_complex_nda_pipeline() -> ExtractionPipeline:
    """Builds a fresh extraction template without source text or gold annotations.

    :return: The reusable contract extraction pipeline.
    :rtype: ExtractionPipeline
    """
    return ExtractionPipeline(
        aspects=[
            Aspect(
                name="NDA information protection",
                description=(
                    "Identify clauses and disclosure-register facts needed to "
                    "determine the protected assets at the Annex A snapshot, "
                    "oral-confirmation timing and its consequences, "
                    "version-specific exclusions, and CEDAR's end of protection. "
                    "Include the governing clock, delivery, duration and "
                    "exclusion rules and relevant asset facts."
                ),
                concepts=[
                    LabelConcept(
                        name="Protected assets at the April snapshot",
                        description=(
                            "Select every complete asset still protected as "
                            "Confidential Information at the Annex A snapshot, "
                            "applying oral-confirmation and all exclusion rules. "
                            "A partial component exclusion does not exclude an "
                            "entire asset."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "ORACLE",
                            "LUMEN",
                            "SHORE",
                            "CEDAR",
                            "MICA",
                            "BASALT",
                        ],
                        classification_type="multi_label",
                    ),
                    DateConcept(
                        name="ORACLE timely confirmation deadline",
                        description=(
                            "Calculate the last date on which written "
                            "confirmation of ORACLE could be delivered in time "
                            "under the original five-Business-Day rule."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    StringConcept(
                        name="ORACLE protection timeline",
                        description=(
                            "Explain whether protection is continuous from the "
                            "oral disclosure through acceptance, identifying the "
                            "protected periods, any gap and the significance of "
                            "sending versus delivery."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    BooleanConcept(
                        name="Entire SHORE package excluded by prior possession",
                        description=(
                            "Determine whether the prior-possession evidence "
                            "excludes the entire disclosed SHORE package from "
                            "confidentiality."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    DateConcept(
                        name="CEDAR end of protection",
                        description=(
                            "Determine the date on which protection of the "
                            "complete CEDAR schedule ends after considering every"
                            " recorded event, rather than reporting a duration "
                            "formula alone."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="NDA recipient permissions",
                description=(
                    "Identify the recipient eligibility, voting-control, "
                    "clean-room and competitor-consent requirements and the "
                    "complete Annex B register needed to decide who may access "
                    "LUMEN and when. Include rules distinguishing executed "
                    "consent from drafts."
                ),
                concepts=[
                    LabelConcept(
                        name=(
                            "People permitted LUMEN access at the September snapshot"
                        ),
                        description=(
                            "Select the people authorised to receive LUMEN at the"
                            " Annex B snapshot after applying all role, "
                            "undertaking, voting-control, clean-room and "
                            "competitor requirements."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "Elin",
                            "Marek",
                            "Nora",
                            "Oskar",
                            "Pia",
                            "Ravi",
                        ],
                        classification_type="multi_label",
                    ),
                    NumericalConcept(
                        name="Talus Research qualifying voting percentage",
                        description=(
                            "Calculate the percentage of Talus Research AB "
                            "general shareholder votes countable toward Affiliate"
                            " status at the access snapshot. Return percentage "
                            "points, not a fraction."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    DateConcept(
                        name="Marek earliest authorised LUMEN access",
                        description=(
                            "Determine the earliest date Marek may access LUMEN, "
                            "applying all relevant approval, employment and "
                            "control conditions to the recorded dates."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    DateConcept(
                        name="Executed competitor consent date for Pia",
                        description=(
                            "Extract the date of an executed Discloser consent "
                            "expressly allowing competitor employee Pia to access"
                            " LUMEN. Return no item if the complete record "
                            "contains no such executed consent."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="NDA compulsory disclosure Q-27",
                description=(
                    "Identify the governing compelled-disclosure scope and notice"
                    " rules, relevant time and delivery definitions, and the "
                    "complete Q-27 order record needed to determine the notice "
                    "deadline, lateness and legally permitted production units."
                ),
                concepts=[
                    DateConcept(
                        name="Q-27 notice deadline date",
                        description=(
                            "Calculate the date of the contractual notice "
                            "deadline to the Discloser for Q-27, applying the "
                            "rule governing an initially prohibited notice."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    NumericalConcept(
                        name="Q-27 notice lateness in minutes",
                        description=(
                            "Calculate how many whole minutes the Q-27 notice "
                            "arrived after its contractual deadline, or zero if "
                            "timely. Apply the actual-arrival exception for "
                            "compelled notices."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        numeric_type="int",
                    ),
                    LabelConcept(
                        name="Q-27 permitted production units",
                        description=(
                            "Select only the production units permitted under the"
                            " final binding scope of Q-27 after narrowing and "
                            "privilege determinations."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "P1",
                            "P2",
                            "P3A",
                            "P3B",
                            "P4",
                        ],
                        classification_type="multi_label",
                    ),
                ],
            ),
            Aspect(
                name="NDA exit obligations X-27",
                description=(
                    "Identify the return and deletion triggers, deadline and "
                    "custody exceptions, residual-knowledge and restricted-use "
                    "rules, and Annex D facts needed to determine active and "
                    "backup deadlines, authorised retention and the proposed "
                    "reuse of NOTE-1. Include the relevant clock and delivery "
                    "definitions."
                ),
                concepts=[
                    DateConcept(
                        name="WORK-1 deletion deadline",
                        description=(
                            "Calculate the deadline for deleting the restored "
                            "WORK-1 active copy, taking account of the request, "
                            "termination, restoration and any preservation facts."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    DateConcept(
                        name="BK-1 deletion deadline",
                        description=(
                            "Calculate the latest permitted deletion date for "
                            "immutable backup BK-1, applying all competing "
                            "deadlines and the restoration facts."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                    ),
                    LabelConcept(
                        name=("Artifacts permitted retention at the December snapshot"),
                        description=(
                            "Select the artifacts permitted to remain under the "
                            "archival, backup or preservation exceptions at the "
                            "Annex D custody snapshot. Do not include ordinary "
                            "copies retained in breach or claimed residual notes."
                        ),
                        add_justifications=True,
                        add_references=True,
                        singular_occurrence=True,
                        labels=[
                            "ARCH-1",
                            "BK-1",
                            "WORK-1",
                            "HOLD-1",
                            "NOTE-1",
                        ],
                        classification_type="multi_label",
                    ),
                    BooleanConcept(
                        name=("NOTE-1 model training permitted as residual knowledge"),
                        description=(
                            "Determine whether the analyst may retain and use "
                            "NOTE-1 to train the proposed diagnostic model under "
                            "the residual-knowledge or retention clauses."
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
                name="NDA effective date",
                description=(
                    "Extract the contractual Effective Date, distinguishing it "
                    "from execution."
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            StringConcept(
                name="NDA permitted purpose",
                description=(
                    "Summarise the sole Permitted Purpose with the transaction "
                    "type, business and project name."
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            DateConcept(
                name="NDA disclosure window end",
                description=(
                    "Extract the actual end of the disclosure window after "
                    "applying the executed notices in the record."
                ),
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            StringConcept(
                name="NDA contractual forum",
                description="Identify the contractual court forum.",
                add_justifications=True,
                add_references=True,
                singular_occurrence=True,
            ),
            JsonObjectConcept(
                name="NDA signatory entities",
                description=(
                    "Extract one object per party with its exact legal name, "
                    "registration country and contractual role. Exclude "
                    "individuals and group companies that are not parties."
                ),
                add_justifications=True,
                add_references=True,
                structure={"name": str, "country": str, "role": str},
            ),
        ],
    )
