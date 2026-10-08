#
# ContextGem
#
# Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
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

"""
Module defining tests for decision models.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from contextgem.internal.decisions import (
    _build_aspect_questions,
    _build_concept_questions,
    _convert_aspect_answer,
    _convert_concept_answers,
    _DecisionAnswer,
    _DecisionOption,
    _DecisionQuestion,
)
from contextgem.internal.exceptions import _InvalidDecisionResponseError
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    JsonObjectConcept,
    LabelConcept,
    NumericalConcept,
    RatingConcept,
    StringConcept,
)
from tests.base import BaseTest


def _predicate_answer(
    name: str, probability: Any = None, **kwargs: Any
) -> _DecisionAnswer:
    """
    Creates the answer to a predicate.

    :param name: Name of the question.
    :type name: str
    :param probability: Probability of "yes", of any type, so that an invalid
        probability can be given as well. Defaults to None.
    :type probability: Any
    :param kwargs: Other params of the answer.
    :type kwargs: Any
    :return: Answer to the predicate.
    :rtype: _DecisionAnswer
    """
    return _DecisionAnswer(
        name=name, kind="predicate", probability=probability, **kwargs
    )


def _choice_answer(
    name: str, option_probabilities: Any = None, **kwargs: Any
) -> _DecisionAnswer:
    """
    Creates the answer to a "choice" question.

    :param name: Name of the question.
    :type name: str
    :param option_probabilities: Probabilities of the options, of any type, so that
        invalid probabilities can be given as well. Defaults to None.
    :type option_probabilities: Any
    :param kwargs: Other params of the answer.
    :type kwargs: Any
    :return: Answer to the "choice" question.
    :rtype: _DecisionAnswer
    """
    return _DecisionAnswer(
        name=name, kind="choice", option_probabilities=option_probabilities, **kwargs
    )


def _score_answer(
    name: str, option_probabilities: Any = None, **kwargs: Any
) -> _DecisionAnswer:
    """
    Creates the answer to a "score" question.

    :param name: Name of the question.
    :type name: str
    :param option_probabilities: Probabilities of the levels, of any type, so that
        invalid probabilities can be given as well. Defaults to None.
    :type option_probabilities: Any
    :param kwargs: Other params of the answer.
    :type kwargs: Any
    :return: Answer to the "score" question.
    :rtype: _DecisionAnswer
    """
    return _DecisionAnswer(
        name=name, kind="score", option_probabilities=option_probabilities, **kwargs
    )


def _check_invalid_answer(
    convert: Callable[[dict[str, _DecisionAnswer]], Any],
    answers: dict[str, _DecisionAnswer],
    question_name: str,
    reason_match: str,
) -> None:
    """
    Checks that the conversion of the answers is rejected as an invalid response
    of the decision model, with an error that names the question.

    :param convert: Function that converts the answers.
    :type convert: Callable[[dict[str, _DecisionAnswer]], Any]
    :param answers: Answers by the names of the questions.
    :type answers: dict[str, _DecisionAnswer]
    :param question_name: Name of the question whose answer is invalid.
    :type question_name: str
    :param reason_match: Pattern that the reason given in the error is to match.
    :type reason_match: str
    """
    with pytest.raises(_InvalidDecisionResponseError, match=reason_match) as exc_info:
        convert(answers)
    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.question_name == question_name
    assert str(exc_info.value).startswith(
        f"Invalid answer to the decision question `{question_name}`: "
    )


# Values that are not a probability, i.e. not a number within [0, 1]
_INVALID_PROBABILITIES: list[Any] = [
    -0.01,
    1.01,
    2,
    float("nan"),
    float("inf"),
    "0.9",
    True,
]


class TestDecisionModels(BaseTest):
    """
    Tests for decision models: the questions built for concepts and aspects, and
    the conversion of the answers to the values of extracted items with their
    probabilities. No requests are made, i.e. no cassettes are required.
    """

    def test_decision_types(self):
        """
        Tests the question and answer types.
        """
        option = _DecisionOption(value="written", label="written")
        assert option.description is None
        question = _DecisionQuestion(
            name="C0", kind="predicate", instructions="Is the document an NDA?"
        )
        assert question.options == ()
        answer = _DecisionAnswer(name="C0", kind="predicate")
        assert answer.probability is None
        assert answer.option_probabilities is None
        assert answer.refused is False

        # Instances are immutable
        for instance, attr_name in [
            (option, "label"),
            (question, "instructions"),
            (answer, "probability"),
        ]:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(instance, attr_name, None)

    def test_build_concept_questions(self):
        """
        Tests the questions built for concepts of each supported type, including
        the names of the questions of several concepts in one sequence.
        """
        concepts = [
            BooleanConcept(
                name="Termination for convenience",
                description="Either party may terminate without cause",
            ),
            LabelConcept(
                name="Notice form",
                description="Required form of the termination notice",
                labels=["written", "email", "any", "not specified"],
            ),
            LabelConcept(
                name="Topics",
                description="Topics covered by the contract",
                labels=["liability", "payment", "termination"],
                classification_type="multi_label",
            ),
            RatingConcept(
                name="Drafting quality",
                description="Overall drafting quality of the contract",
                rating_scale=(0, 4),
            ),
            BooleanConcept(name="Is a contract", description="Document is a contract"),
            RatingConcept(
                name="Risk level",
                description="Level of risk for the receiving party",
                rating_scale=(3, 7),
            ),
        ]
        questions = _build_concept_questions(concepts)
        assert all(isinstance(question, _DecisionQuestion) for question in questions)
        assert [(question.name, question.kind) for question in questions] == [
            ("C0", "predicate"),
            ("C1", "choice"),
            ("C2-L0", "predicate"),
            ("C2-L1", "predicate"),
            ("C2-L2", "predicate"),
            ("C3", "score"),
            ("C4", "predicate"),
            ("C5", "score"),
        ]
        questions_by_name = {question.name: question for question in questions}
        assert len(questions_by_name) == len(questions)
        for question in questions:
            assert question.instructions == question.instructions.strip()
            self.check_rendered_prompt(question.instructions)
            if question.kind == "predicate":
                assert question.options == ()

        # Boolean concept
        assert questions_by_name["C0"].instructions == (
            "Decide whether the statement in the concept description below is true "
            "according to the text.\n"
            "Concept name: Termination for convenience\n"
            "Concept description: Either party may terminate without cause\n"
            'Answer "yes" if the text supports the statement. Answer "no" if '
            "the text contradicts the statement or does not support it."
        )
        assert questions_by_name["C4"].instructions == (
            "Decide whether the statement in the concept description below is true "
            "according to the text.\n"
            "Concept name: Is a contract\n"
            "Concept description: Document is a contract\n"
            'Answer "yes" if the text supports the statement. Answer "no" if '
            "the text contradicts the statement or does not support it."
        )

        # Label concept, a single label to select
        assert questions_by_name["C1"].instructions == (
            "Classify the text by selecting exactly one of the options, based on "
            "the concept below.\n"
            "Concept name: Notice form\n"
            "Concept description: Required form of the termination notice\n"
            "Select the most appropriate option, even if no option fits the text "
            "perfectly."
        )
        assert questions_by_name["C1"].options == (
            _DecisionOption(value="written", label="written", description=None),
            _DecisionOption(value="email", label="email", description=None),
            _DecisionOption(value="any", label="any", description=None),
            _DecisionOption(
                value="not specified", label="not specified", description=None
            ),
        )

        # Label concept, any number of labels to select: a question for each label
        for label_position, label in enumerate(["liability", "payment", "termination"]):
            assert questions_by_name[f"C2-L{label_position}"].instructions == (
                "Decide whether the label below applies to the text, based on "
                "the concept below.\n"
                "Concept name: Topics\n"
                "Concept description: Topics covered by the contract\n"
                f"Label: {label}\n"
                'Answer "yes" if the label applies to the text. Answer "no" if it '
                "does not. The other labels of the concept are decided separately: "
                "several labels, or none, may apply."
            )

        # Rating concept, including the scales that do not start at 1
        assert questions_by_name["C3"].instructions == (
            "Rate the text on a scale from 0 to 4, based on the concept below.\n"
            "Concept name: Drafting quality\n"
            "Concept description: Overall drafting quality of the contract\n"
            "Select the level of the scale that fits the text best."
        )
        assert questions_by_name["C3"].options == (
            _DecisionOption(value=0, label="0", description="0 on a scale from 0 to 4"),
            _DecisionOption(value=1, label="1", description="1 on a scale from 0 to 4"),
            _DecisionOption(value=2, label="2", description="2 on a scale from 0 to 4"),
            _DecisionOption(value=3, label="3", description="3 on a scale from 0 to 4"),
            _DecisionOption(value=4, label="4", description="4 on a scale from 0 to 4"),
        )
        assert questions_by_name["C5"].instructions.startswith(
            "Rate the text on a scale from 3 to 7, based on the concept below.\n"
        )
        assert [
            (option.value, option.label, option.description)
            for option in questions_by_name["C5"].options
        ] == [
            (level, str(level), f"{level} on a scale from 3 to 7")
            for level in [3, 4, 5, 6, 7]
        ]
        assert all(
            type(option.value) is int
            for name in ["C3", "C5"]
            for option in questions_by_name[name].options
        )

        # The name of a question follows the position of the concept in the sequence
        assert [
            question.name for question in _build_concept_questions(concepts[2:4])
        ] == ["C0-L0", "C0-L1", "C0-L2", "C1"]

        # No concepts, no questions
        assert _build_concept_questions([]) == []

    def test_build_aspect_questions(self):
        """
        Tests the questions built for aspects.
        """
        aspects = [
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                aspects=[
                    Aspect(
                        name="Liability cap",
                        description="Clauses that cap the liability of a party",
                    )
                ],
            ),
            Aspect(
                name="Termination",
                description="Clauses on termination of the agreement",
            ),
        ]
        questions = _build_aspect_questions(aspects)
        # A question for each aspect of the sequence, i.e. not for the sub-aspects
        assert questions == [
            _DecisionQuestion(
                name="A0",
                kind="predicate",
                instructions=(
                    "The text is one paragraph of a document. Decide whether "
                    "the paragraph is relevant to the aspect below.\n"
                    "Aspect name: Liability\n"
                    "Aspect description: Clauses describing liability of the parties\n"
                    'Answer "yes" if the paragraph addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
            _DecisionQuestion(
                name="A1",
                kind="predicate",
                instructions=(
                    "The text is one paragraph of a document. Decide whether "
                    "the paragraph is relevant to the aspect below.\n"
                    "Aspect name: Termination\n"
                    "Aspect description: Clauses on termination of the agreement\n"
                    'Answer "yes" if the paragraph addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
        ]
        for question in questions:
            self.check_rendered_prompt(question.instructions)

        # The name of a question follows the position of the aspect in the sequence
        assert [question.name for question in _build_aspect_questions(aspects[1:])] == [
            "A0"
        ]

        # No aspects, no questions
        assert _build_aspect_questions([]) == []

        # Only aspects get the questions of aspects
        not_only_aspects: list[Any] = [
            aspects[0],
            BooleanConcept(name="Is a contract", description="Document is a contract"),
        ]
        with pytest.raises(
            TypeError, match="Expected an aspect .* got `BooleanConcept`"
        ):
            _build_aspect_questions(not_only_aspects)

    def test_unsupported_concept_types(self):
        """
        Tests that the concepts of the types that a decision model cannot extract
        are rejected with an error that names the concept.
        """
        boolean_concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        unsupported_concepts = [
            StringConcept(name="Contract title", description="Title of the contract"),
            NumericalConcept(name="Notice period", description="Notice period in days"),
            DateConcept(name="Effective date", description="Effective date"),
            JsonObjectConcept(
                name="Party details",
                description="Details of a party",
                structure={"name": str},
            ),
        ]
        for concept in unsupported_concepts:
            error_match = (
                f"Concept `{concept.name}` of type `{type(concept).__name__}` "
                f"cannot be extracted by a decision model"
            )
            # Also when the other concepts in the sequence are supported
            for concepts in [[concept], [boolean_concept, concept]]:
                with pytest.raises(TypeError, match=error_match):
                    _build_concept_questions(concepts)
            with pytest.raises(TypeError, match=error_match):
                _convert_concept_answers(
                    concept, 0, {"C0": _predicate_answer("C0", probability=0.9)}
                )

    def test_convert_boolean_concept_answers(self):
        """
        Tests the conversion of the answer to the question of a Boolean concept,
        including the probability of exactly 0.5.
        """
        concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        for probability, expected_value in [
            (0.93, True),
            (0.5, True),
            (0.4999, False),
            (0.07, False),
            (0.0, False),
            (1.0, True),
            (0, False),
            (1, True),
        ]:
            result = _convert_concept_answers(
                concept, 0, {"C0": _predicate_answer("C0", probability=probability)}
            )
            assert result is not None
            value, probabilities = result
            assert value is expected_value
            assert probabilities == pytest.approx(
                {True: probability, False: 1 - probability}
            )
            assert [type(option) for option in probabilities] == [bool, bool]
            assert list(probabilities) == [True, False]
            assert all(type(i) is float for i in probabilities.values())

        # The answer is taken by the position of the concept, and the answers to
        # the other questions of the request are not read
        answers = {
            "C0": _predicate_answer("C0", probability=0.9),
            "C1": _predicate_answer("C1", probability=0.2),
            "C2": _choice_answer("C2", option_probabilities={"a": 0.6, "b": 0.4}),
            "A1": _predicate_answer("A1", probability=0.8),
        }
        assert _convert_concept_answers(concept, 0, answers) == (
            True,
            pytest.approx({True: 0.9, False: 0.1}),
        )
        assert _convert_concept_answers(concept, 1, answers) == (
            False,
            pytest.approx({True: 0.2, False: 0.8}),
        )

    def test_convert_multi_class_label_concept_answers(self):
        """
        Tests the conversion of the answer to the question of a label concept
        with a single label to select, including the equally probable labels and
        the probabilities that do not sum to 1.
        """
        labels = ["written", "email", "any", "not specified"]
        concept = LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=labels,
        )
        assert concept.classification_type == "multi_class"

        for option_probabilities, expected_label in [
            # The most probable label
            (
                {"written": 0.1, "email": 0.7, "any": 0.15, "not specified": 0.05},
                "email",
            ),
            (
                {"written": 0.0, "email": 0.0, "any": 0.0, "not specified": 1.0},
                "not specified",
            ),
            # The probabilities given in another order than the labels of the concept
            (
                {"not specified": 0.05, "any": 0.15, "email": 0.7, "written": 0.1},
                "email",
            ),
            # Equally probable labels: the first of them in the order of the labels
            # of the concept, not in the order of the answer
            (
                {"written": 0.1, "email": 0.4, "any": 0.4, "not specified": 0.1},
                "email",
            ),
            (
                {"not specified": 0.3, "any": 0.3, "email": 0.1, "written": 0.3},
                "written",
            ),
            (
                {"written": 0.25, "email": 0.25, "any": 0.25, "not specified": 0.25},
                "written",
            ),
            (
                {"written": 0.0, "email": 0.0, "any": 0.0, "not specified": 0.0},
                "written",
            ),
            # The probabilities that do not sum to 1 are taken as they are
            (
                {"written": 0.1, "email": 0.1, "any": 0.3, "not specified": 0.1},
                "any",
            ),
            (
                {"written": 0.9, "email": 0.8, "any": 1.0, "not specified": 0.7},
                "any",
            ),
            ({"written": 1, "email": 0, "any": 0, "not specified": 0}, "written"),
        ]:
            result = _convert_concept_answers(
                concept,
                3,
                {"C3": _choice_answer("C3", option_probabilities=option_probabilities)},
            )
            assert result is not None
            value, probabilities = result
            assert value == [expected_label]
            # Each label of the concept with its probability as given, in the order
            # of the labels of the concept
            assert probabilities == option_probabilities
            assert list(probabilities) == labels
            assert all(type(i) is float for i in probabilities.values())

    def test_convert_multi_label_label_concept_answers(self):
        """
        Tests the conversion of the answers to the questions of a label concept
        with any number of labels to select, including the probability of exactly
        0.5 and no label selected.
        """
        labels = ["liability", "payment", "termination"]
        concept = LabelConcept(
            name="Topics",
            description="Topics covered by the contract",
            labels=labels,
            classification_type="multi_label",
        )

        for label_probabilities, expected_labels in [
            ([0.9, 0.2, 0.8], ["liability", "termination"]),
            ([0.1, 0.6, 0.3], ["payment"]),
            # All labels: the probabilities of the labels are independent
            ([1.0, 0.95, 0.5], ["liability", "payment", "termination"]),
            # A label with the probability of exactly 0.5 is selected
            ([0.5, 0.4999, 0.0], ["liability"]),
            ([0.4999, 0.5, 0.5], ["payment", "termination"]),
        ]:
            # The answers are given in the reverse order of the labels
            answers = {
                f"C1-L{label_position}": _predicate_answer(
                    f"C1-L{label_position}", probability=probability
                )
                for label_position, probability in reversed(
                    list(enumerate(label_probabilities))
                )
            }
            result = _convert_concept_answers(concept, 1, answers)
            assert result is not None
            value, probabilities = result
            # The selected labels are in the order of the labels of the concept
            assert value == expected_labels
            # Each label of the concept with the probability of its own question
            assert probabilities == dict(zip(labels, label_probabilities, strict=True))
            assert list(probabilities) == labels

        # No label is selected: no item
        for label_probabilities in [[0.4999, 0.1, 0.0], [0.0, 0.0, 0.0]]:
            answers = {
                f"C1-L{label_position}": _predicate_answer(
                    f"C1-L{label_position}", probability=probability
                )
                for label_position, probability in enumerate(label_probabilities)
            }
            assert _convert_concept_answers(concept, 1, answers) is None

    def test_convert_rating_concept_answers(self):
        """
        Tests the conversion of the answer to the question of a rating concept,
        including the scales that do not start at 1 and the equally probable levels.
        """
        for rating_scale, option_probabilities, expected_value in [
            # The most probable level
            ((1, 5), {"1": 0.05, "2": 0.1, "3": 0.15, "4": 0.6, "5": 0.1}, 4),
            ((1, 5), {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 1.0}, 5),
            ((0, 4), {"0": 0.05, "1": 0.1, "2": 0.6, "3": 0.2, "4": 0.05}, 2),
            # The lowest level of a scale that starts at 0, i.e. the value 0 is
            # an extracted value
            ((0, 4), {"0": 0.5, "1": 0.3, "2": 0.1, "3": 0.05, "4": 0.05}, 0),
            ((3, 7), {"3": 0.1, "4": 0.1, "5": 0.1, "6": 0.1, "7": 0.6}, 7),
            ((3, 7), {"3": 0.7, "4": 0.1, "5": 0.1, "6": 0.05, "7": 0.05}, 3),
            ((0, 1), {"0": 0.2, "1": 0.8}, 1),
            # The probabilities given in another order than the levels of the scale
            ((3, 7), {"7": 0.1, "5": 0.6, "3": 0.1, "6": 0.1, "4": 0.1}, 5),
            # Equally probable levels: the lowest of them
            ((1, 5), {"1": 0.1, "2": 0.3, "3": 0.3, "4": 0.3, "5": 0.0}, 2),
            ((3, 7), {"7": 0.4, "6": 0.1, "5": 0.1, "4": 0.4, "3": 0.0}, 4),
            ((0, 4), {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}, 0),
            # The probabilities that do not sum to 1 are taken as they are
            ((0, 4), {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.2, "4": 0.1}, 3),
            ((3, 7), {"3": 0.9, "4": 1.0, "5": 0.9, "6": 0.8, "7": 0.7}, 4),
        ]:
            concept = RatingConcept(
                name="Drafting quality",
                description="Overall drafting quality of the contract",
                rating_scale=rating_scale,
            )
            result = _convert_concept_answers(
                concept,
                2,
                {"C2": _score_answer("C2", option_probabilities=option_probabilities)},
            )
            assert result is not None
            value, probabilities = result
            assert type(value) is int
            assert value == expected_value
            # Each level of the scale with its probability as given, with
            # the integers of the scale as keys, in ascending order
            levels = list(range(rating_scale[0], rating_scale[1] + 1))
            assert probabilities == {
                level: option_probabilities[str(level)] for level in levels
            }
            assert list(probabilities) == levels
            assert all(type(level) is int for level in probabilities)

    def test_convert_aspect_answer(self):
        """
        Tests the conversion of the answer to the question of an aspect, asked on
        one paragraph, including the probability of exactly 0.5.
        """
        for probability, expected_belongs in [
            (0.93, True),
            (0.5, True),
            (0.4999, False),
            (0.0, False),
            (1.0, True),
            (1, True),
        ]:
            belongs, probabilities = _convert_aspect_answer(
                0, {"A0": _predicate_answer("A0", probability=probability)}
            )
            assert belongs is expected_belongs
            assert probabilities == pytest.approx(
                {True: probability, False: 1 - probability}
            )
            assert [type(option) for option in probabilities] == [bool, bool]
            assert list(probabilities) == [True, False]

        # The answer is taken by the position of the aspect, and the answers to
        # the other questions of the request are not read
        answers = {
            "A0": _predicate_answer("A0", probability=0.1),
            "A1": _predicate_answer("A1", probability=0.7),
            "C1": _predicate_answer("C1", probability=0.0),
        }
        assert _convert_aspect_answer(0, answers) == (
            False,
            pytest.approx({True: 0.1, False: 0.9}),
        )
        assert _convert_aspect_answer(1, answers) == (
            True,
            pytest.approx({True: 0.7, False: 0.3}),
        )

    def test_invalid_predicate_answers(self):
        """
        Tests that an answer to a predicate that cannot be used is rejected as
        an invalid response, for a Boolean concept, for a label of a label concept
        and for an aspect.
        """
        boolean_concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        multi_label_concept = LabelConcept(
            name="Topics",
            description="Topics covered by the contract",
            labels=["liability", "payment", "termination"],
            classification_type="multi_label",
        )
        # Valid answers to the questions of the labels
        multi_label_answers = {
            f"C1-L{label_position}": _predicate_answer(
                f"C1-L{label_position}", probability=0.8
            )
            for label_position in range(3)
        }
        assert _convert_concept_answers(multi_label_concept, 1, multi_label_answers)

        def convert_boolean(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the Boolean concept at position 0.
            """
            return _convert_concept_answers(boolean_concept, 0, answers)

        def convert_multi_label(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the label concept at position 1, with
            the given answers used instead of the valid ones.
            """
            return _convert_concept_answers(
                multi_label_concept, 1, {**multi_label_answers, **answers}
            )

        def convert_aspect(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the aspect at position 2.
            """
            return _convert_aspect_answer(2, answers)

        for convert, name in [
            (convert_boolean, "C0"),
            # One answer that cannot be used makes the response invalid, even if
            # the answers to the questions of the other labels are valid
            (convert_multi_label, "C1-L1"),
            (convert_aspect, "A2"),
        ]:
            # Refused
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name, refused=True)},
                name,
                "the model refused to answer",
            )
            # Refused, even if answered as well
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name, probability=0.9, refused=True)},
                name,
                "the model refused to answer",
            )
            # Of another kind than the question
            _check_invalid_answer(
                convert,
                {name: _choice_answer(name, option_probabilities={"yes": 0.9})},
                name,
                "the answer is of the kind `choice`, while the question is of "
                "the kind `predicate`",
            )
            _check_invalid_answer(
                convert,
                {name: _score_answer(name, option_probabilities={"1": 0.9})},
                name,
                "the answer is of the kind `score`, while the question is of "
                "the kind `predicate`",
            )
            # Without the probability
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name)},
                name,
                "the answer has no probability",
            )
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name, option_probabilities={"yes": 0.9})},
                name,
                "the answer has no probability",
            )
            # With a probability that is not a number within [0, 1]
            for probability in _INVALID_PROBABILITIES:
                _check_invalid_answer(
                    convert,
                    {name: _predicate_answer(name, probability=probability)},
                    name,
                    r"is not a number within \[0, 1\]",
                )

        # Missing
        for answers in [{}, {"C1": _predicate_answer("C1", probability=0.9)}]:
            _check_invalid_answer(
                convert_boolean, answers, "C0", "the answer is missing"
            )
        _check_invalid_answer(
            lambda answers: _convert_concept_answers(multi_label_concept, 1, answers),
            {
                name: answer
                for name, answer in multi_label_answers.items()
                if name != "C1-L2"
            },
            "C1-L2",
            "the answer is missing",
        )
        for answers in [{}, {"A0": _predicate_answer("A0", probability=0.9)}]:
            _check_invalid_answer(
                convert_aspect, answers, "A2", "the answer is missing"
            )

    def test_invalid_choice_and_score_answers(self):
        """
        Tests that an answer to a "choice" or a "score" question that cannot be
        used is rejected as an invalid response, for a label concept with a single
        label to select and for a rating concept.
        """
        multi_class_concept = LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=["written", "email", "any"],
        )
        rating_concept = RatingConcept(
            name="Drafting quality",
            description="Overall drafting quality of the contract",
            rating_scale=(3, 5),
        )

        def convert_multi_class(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the label concept at position 0.
            """
            return _convert_concept_answers(multi_class_concept, 0, answers)

        def convert_rating(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the rating concept at position 4.
            """
            return _convert_concept_answers(rating_concept, 4, answers)

        for convert, name, kind, create_answer, option_labels, other_labels in [
            (
                convert_multi_class,
                "C0",
                "choice",
                _choice_answer,
                ["written", "email", "any"],
                # Labels that are not the labels of the concept, including
                # a label in another case
                ["other", "Written", "0"],
            ),
            (
                convert_rating,
                "C4",
                "score",
                _score_answer,
                ["3", "4", "5"],
                # Levels that are not the levels of the scale, including the level
                # positions and the levels given in another format
                ["2", "6", "0", "3.0", "three"],
            ),
        ]:
            valid_probabilities = dict(zip(option_labels, [0.2, 0.5, 0.3], strict=True))
            assert convert(
                {name: create_answer(name, option_probabilities=valid_probabilities)}
            )

            # Missing
            for answers in [
                {},
                {"C1": create_answer("C1", option_probabilities=valid_probabilities)},
            ]:
                _check_invalid_answer(convert, answers, name, "the answer is missing")
            # Refused
            _check_invalid_answer(
                convert,
                {name: create_answer(name, refused=True)},
                name,
                "the model refused to answer",
            )
            # Refused, even if answered as well
            _check_invalid_answer(
                convert,
                {
                    name: create_answer(
                        name, option_probabilities=valid_probabilities, refused=True
                    )
                },
                name,
                "the model refused to answer",
            )
            # Of another kind than the question
            for other_answer in [
                _predicate_answer(name, probability=0.9),
                _DecisionAnswer(
                    name=name,
                    kind="score" if kind == "choice" else "choice",
                    option_probabilities=valid_probabilities,
                ),
            ]:
                _check_invalid_answer(
                    convert,
                    {name: other_answer},
                    name,
                    f"the answer is of the kind `{other_answer.kind}`, while "
                    f"the question is of the kind `{kind}`",
                )
            # Without the probabilities of the options
            for answer in [
                create_answer(name),
                create_answer(name, option_probabilities={}),
                create_answer(name, probability=0.9),
            ]:
                _check_invalid_answer(
                    convert,
                    {name: answer},
                    name,
                    "the answer has no probabilities of the options",
                )
            # With an option that was not offered, in addition to the options
            # offered or instead of one of them
            for other_label in other_labels:
                for option_probabilities in [
                    {**valid_probabilities, other_label: 0.1},
                    {
                        **dict(zip(option_labels[:2], [0.2, 0.5], strict=True)),
                        other_label: 0.3,
                    },
                    {other_label: 1.0},
                ]:
                    _check_invalid_answer(
                        convert,
                        {
                            name: create_answer(
                                name, option_probabilities=option_probabilities
                            )
                        },
                        name,
                        "the answer has options that were not offered: "
                        rf"\['{other_label}'\]",
                    )
            # Without one of the options offered: an option without a probability
            # is not taken as an option with a zero probability
            for omitted_label in option_labels:
                _check_invalid_answer(
                    convert,
                    {
                        name: create_answer(
                            name,
                            option_probabilities={
                                label: probability
                                for label, probability in valid_probabilities.items()
                                if label != omitted_label
                            },
                        )
                    },
                    name,
                    "the answer has no probabilities of the options "
                    rf"\['{omitted_label}'\]",
                )
            # With a probability that is not a number within [0, 1]
            for probability in [*_INVALID_PROBABILITIES, None]:
                _check_invalid_answer(
                    convert,
                    {
                        name: create_answer(
                            name,
                            option_probabilities={
                                **valid_probabilities,
                                option_labels[1]: probability,
                            },
                        )
                    },
                    name,
                    rf"of the option `{option_labels[1]}` is not a number within "
                    r"\[0, 1\]",
                )
