# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the questions and answers of decision models.

A decision model takes a text and a set of typed questions (yes/no, pick one option,
rate on a scale) and returns probabilities for each question. It generates no text.

The module defines the question and answer types, which are independent of the wire
formats of the decision model APIs, the mapping from concepts and aspects to
questions, the text of a paragraph or of a sentence that the questions of aspects
are asked on, and the conversion of answers to the values of extracted items with
their probability distributions. The functions are pure: the module sends no
requests and does not depend on the library that sends them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast

from jinja2 import Template

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.concepts import (
    _BooleanConcept,
    _Concept,
    _LabelConcept,
    _RatingConcept,
)
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.exceptions import _InvalidDecisionResponseError
from contextgem.internal.utils import _clean_text_for_llm_prompt, _get_template


# Kind of a decision question, and of the answer to it: a yes/no question,
# a question with one option to pick, or a question with a level to rate on.
_DecisionKind = Literal["predicate", "choice", "score"]

# What a question of an aspect is asked on, i.e. what is judged for belonging to
# the aspect: a paragraph, or a sentence of a paragraph that belongs to the aspect,
# for an aspect that is extracted with sentences as its items.
_AspectTextUnit = Literal["paragraph", "sentence"]

# Value of an extracted item, as converted from the answers of a decision model:
# a boolean (Boolean concept), an integer of the rating scale (rating concept),
# or a list of labels (label concept).
_DecisionItemValue = bool | int | list[str]

# Probability distribution of an extracted item: maps each option, as it appears
# as an item value (``True`` / ``False``, a label, an integer of the rating scale),
# to its probability.
_DecisionProbabilities = dict[bool | int | str, float]

# Default decision threshold: the lowest probability of "yes" with which a yes/no
# question is taken as answered "yes", unless another threshold is set for
# the decision model, or for the aspect or the concept that the question is of
_DEFAULT_DECISION_THRESHOLD = 0.5

# Names of the templates of the question instructions
_CONCEPT_QUESTION_TEMPLATE_NAME = "decision_question_concept"
_ASPECT_QUESTION_TEMPLATE_NAME = "decision_question_aspect"


@dataclass(frozen=True)
class _DecisionOption:
    """
    Option of a decision question: an option of a "choice" question, or a level of
    a "score" question.

    :ivar value: What the option stands for on an extracted item: a label of a label
        concept, or an integer of the scale of a rating concept.
    :vartype value: str | int
    :ivar label: Text of the option sent to the model. Unique within its question.
    :vartype label: str
    :ivar description: Description of the option sent to the model with its label,
        if any. Defaults to None.
    :vartype description: str | None
    """

    value: str | int
    label: str
    description: str | None = None


@dataclass(frozen=True)
class _DecisionQuestion:
    """
    Question to a decision model, to be answered on a text.

    :ivar name: Name of the question, unique within one request. The answer to
        the question has the same name.
    :vartype name: str
    :ivar kind: Kind of the question: "predicate" (a yes/no question), "choice"
        (one of the options is to be picked), or "score" (the text is to be rated on
        the levels given as the options, in ascending order).
    :vartype kind: Literal["predicate", "choice", "score"]
    :ivar instructions: Text of the question sent to the model.
    :vartype instructions: str
    :ivar options: Options of a "choice" question, or levels of a "score" question,
        in the order in which they are sent to the model. Empty for a predicate.
    :vartype options: tuple[_DecisionOption, ...]
    """

    name: str
    kind: _DecisionKind
    instructions: str
    options: tuple[_DecisionOption, ...] = ()


@dataclass(frozen=True)
class _DecisionAnswer:
    """
    Answer of a decision model to a question.

    An answer is not validated when it is created: it is validated when it is
    converted (see :func:`_convert_concept_answers` and
    :func:`_convert_aspect_answer`), so that an answer that cannot be used is
    reported with the name of its question.

    :ivar name: Name of the question that is answered.
    :vartype name: str
    :ivar kind: Kind of the answer, which must be the kind of the question:
        "predicate", "choice" or "score".
    :vartype kind: Literal["predicate", "choice", "score"]
    :ivar probability: Probability of "yes", for the answer to a predicate.
        Defaults to None.
    :vartype probability: float | None
    :ivar option_probabilities: Probability of each option of the question, by
        the label of the option (see :attr:`_DecisionOption.label`), for the answer
        to a "choice" or a "score" question. Defaults to None.
    :vartype option_probabilities: Mapping[str, float] | None
    :ivar refused: Whether the model refused to answer the question. A refused
        answer tells nothing, whatever its kind and probabilities are: the text
        that the question of an aspect was asked on is then not included in
        the aspect (see :func:`_convert_aspect_answer`), while a refused answer
        to a question of a concept cannot be used. Defaults to False.
    :vartype refused: bool
    :ivar chosen_option: Label of the option that the model picked, for the answer
        to a "choice" question that names it. It must be one of the options of
        the question. Defaults to None, i.e. the picked option is not named.
    :vartype chosen_option: str | bool | None
    """

    name: str
    kind: _DecisionKind
    probability: float | None = None
    option_probabilities: Mapping[str, float] | None = None
    refused: bool = False
    chosen_option: str | bool | None = None


# Building questions


def _build_concept_questions(concepts: Sequence[_Concept]) -> list[_DecisionQuestion]:
    """
    Builds the questions that a decision model is to answer on a text to extract
    the concepts.

    The names of the questions are built from the position ``i`` of the concept in
    the sequence, so that they are unique within the request for these concepts:

    - Boolean concept: a predicate ``C{i}``;
    - label concept with the "multi_class" classification type: a "choice" question
      ``C{i}``, with one option for each label of the concept;
    - label concept with the "multi_label" classification type: a predicate
      ``C{i}-L{j}`` for each label of the concept, where ``j`` is the position of
      the label in the labels of the concept;
    - rating concept: a "score" question ``C{i}``, with one level for each integer
      of the rating scale, in ascending order.

    The answers to the questions are converted to the values of the extracted items
    with :func:`_convert_concept_answers`, called for each concept with the same
    position.

    :param concepts: Concepts to build the questions for. Only Boolean, label and
        rating concepts are supported.
    :type concepts: Sequence[_Concept]
    :return: Questions in the order of the concepts and, for a label concept with
        the "multi_label" classification type, of its labels.
    :rtype: list[_DecisionQuestion]
    :raises TypeError: If a concept is of a type that a decision model
        cannot extract.
    """
    questions: list[_DecisionQuestion] = []
    for position, concept in enumerate(concepts):
        if isinstance(concept, _BooleanConcept):
            questions.append(
                _DecisionQuestion(
                    name=_get_concept_question_name(position),
                    kind="predicate",
                    instructions=_render_concept_question_instructions(
                        concept, question_type="boolean"
                    ),
                )
            )
        elif isinstance(concept, _LabelConcept):
            if concept.classification_type == "multi_class":
                questions.append(
                    _DecisionQuestion(
                        name=_get_concept_question_name(position),
                        kind="choice",
                        instructions=_render_concept_question_instructions(
                            concept, question_type="multi_class"
                        ),
                        options=_get_label_options(concept),
                    )
                )
            else:
                questions.extend(
                    _DecisionQuestion(
                        name=_get_concept_question_name(position, label_position),
                        kind="predicate",
                        instructions=_render_concept_question_instructions(
                            concept, question_type="multi_label", label=label
                        ),
                    )
                    for label_position, label in enumerate(concept.labels)
                )
        elif isinstance(concept, _RatingConcept):
            questions.append(
                _DecisionQuestion(
                    name=_get_concept_question_name(position),
                    kind="score",
                    instructions=_render_concept_question_instructions(
                        concept,
                        question_type="rating",
                        rating_start=concept._rating_start,
                        rating_end=concept._rating_end,
                    ),
                    options=_get_rating_options(concept),
                )
            )
        else:
            raise _get_unsupported_concept_error(concept)
    return questions


def _build_aspect_questions(
    aspects: Sequence[_Aspect],
    has_additional_context: bool = False,
    text_unit: _AspectTextUnit = "paragraph",
) -> list[_DecisionQuestion]:
    """
    Builds the questions that a decision model is to answer on one paragraph to
    extract the aspects: for each aspect, whether the paragraph belongs to it.

    The question of an aspect is a predicate ``A{i}``, where ``i`` is the position
    of the aspect in the sequence, so that the names of the questions are unique
    within the request for these aspects. The questions depend on the paragraph
    only in whether the text that they are asked on has the additional context of
    the paragraph (see :func:`_get_paragraph_text_for_aspects`), which
    the instructions of the questions then tell: the same questions are asked on
    each paragraph with additional context, and the same questions on each
    paragraph without it.

    The questions can also be built for one sentence, for the aspects that are
    extracted with sentences as their items: for each aspect, whether the sentence
    belongs to it. The text that such questions are asked on has the sentence and
    the paragraph that the sentence is from (see
    :func:`_get_sentence_text_for_aspects`), and the instructions tell that
    the sentence is what is judged. The names of the questions are the same as
    for a paragraph, i.e. the questions of some of the aspects only, taken from
    those built for all the aspects of a call, keep the names that tell
    the positions of their aspects in the call.

    Only the aspects of the sequence get a question, i.e. the sub-aspects of
    an aspect do not.

    The answer to a question is converted with :func:`_convert_aspect_answer`,
    called for each aspect with the same position.

    :param aspects: Aspects to build the questions for.
    :type aspects: Sequence[_Aspect]
    :param has_additional_context: Whether the text that the questions are asked
        on has additional context, next to the paragraph: that of the paragraph
        or, for the questions on a sentence, that of the sentence, of its
        paragraph, or of both. Defaults to False, i.e. the text is the paragraph
        alone, or the sentence and its paragraph.
    :type has_additional_context: bool
    :param text_unit: What the questions are asked on, i.e. what is judged:
        "paragraph" or "sentence". Defaults to "paragraph".
    :type text_unit: Literal["paragraph", "sentence"]
    :return: Questions in the order of the aspects, one for each aspect.
    :rtype: list[_DecisionQuestion]
    :raises TypeError: If an object in the sequence is not an aspect.
    """
    questions: list[_DecisionQuestion] = []
    for position, aspect in enumerate(aspects):
        if not isinstance(aspect, _Aspect):
            raise TypeError(
                f"Expected an aspect to build a decision question for, "
                f"got `{type(aspect).__name__}`: {aspect!r}"
            )
        instructions = _render_instructions(
            _ASPECT_QUESTION_TEMPLATE_NAME,
            aspect_name=aspect.name,
            aspect_description=aspect.description,
            has_additional_context=has_additional_context,
            text_unit=text_unit,
        )
        questions.append(
            _DecisionQuestion(
                name=_get_aspect_question_name(position),
                kind="predicate",
                instructions=instructions,
            )
        )
    return questions


def _get_paragraph_text_for_aspects(paragraph: _Paragraph) -> tuple[str, bool]:
    """
    Returns the text of a paragraph that a decision model is to answer
    the questions of aspects on (see :func:`_build_aspect_questions`).

    The text is the text of the paragraph as it is shown to an LLM for
    the extraction of aspects, i.e. cleaned and without line breaks. If
    the paragraph has additional context, e.g. its formatting, list context or
    table position, the text has both, each with a label, so that the model can
    tell the paragraph, which is what it judges, from its context::

        Paragraph: <text of the paragraph>

        Additional context of the paragraph: <additional context>

    The additional context is cleaned as the text of the paragraph is. A paragraph
    without additional context is sent as its text alone, without a label.

    :param paragraph: Paragraph to ask the questions of aspects on.
    :type paragraph: _Paragraph
    :return: The text to ask the questions on, which is empty if the paragraph has
        no text to show to a model, i.e. it is not to be asked about, and whether
        the text has the additional context of the paragraph, i.e. whether
        the questions are to be built for a text with additional context.
    :rtype: tuple[str, bool]
    """
    text = _clean_text_for_question(paragraph.raw_text)
    additional_context = _clean_text_for_question(paragraph.additional_context)
    if not text or not additional_context:
        return text, False
    return (
        (
            f"Paragraph: {text}\n\n"
            f"Additional context of the paragraph: {additional_context}"
        ),
        True,
    )


def _get_sentence_text_for_aspects(
    sentence: _Sentence, paragraph: _Paragraph
) -> tuple[str, bool]:
    """
    Returns the text of a sentence that a decision model is to answer
    the questions of aspects on, built for a sentence (see
    :func:`_build_aspect_questions`).

    A sentence is rarely understood on its own, so the text has the sentence,
    which is what the model judges, and the paragraph that the sentence is from,
    as its context, each with a label. The additional context of the sentence and
    that of the paragraph follow, each with a label, if they are set::

        Sentence: <text of the sentence>

        Paragraph that the sentence is from: <text of the paragraph>

        Additional context of the sentence: <additional context>

        Additional context of the paragraph: <additional context>

    The texts and the additional context are cleaned as the text of a paragraph
    is (see :func:`_get_paragraph_text_for_aspects`). A sentence that is segmented
    from a paragraph inherits the additional context of the paragraph: additional
    context that the sentence and the paragraph share is given once, as that of
    the paragraph.

    :param sentence: Sentence to ask the questions of aspects on.
    :type sentence: _Sentence
    :param paragraph: Paragraph that the sentence is from.
    :type paragraph: _Paragraph
    :return: The text to ask the questions on, which is empty if the sentence has
        no text to show to a model, i.e. it is not to be asked about, and whether
        the text has additional context, of the sentence or of the paragraph, i.e.
        whether the questions are to be built for a text with additional context.
    :rtype: tuple[str, bool]
    """
    sentence_text = _clean_text_for_question(sentence.raw_text)
    if not sentence_text:
        return "", False
    parts = [
        f"Sentence: {sentence_text}",
        (
            "Paragraph that the sentence is from: "
            f"{_clean_text_for_question(paragraph.raw_text)}"
        ),
    ]
    sentence_context = _clean_text_for_question(sentence.additional_context)
    paragraph_context = _clean_text_for_question(paragraph.additional_context)
    if sentence_context and sentence_context != paragraph_context:
        parts.append(f"Additional context of the sentence: {sentence_context}")
    if paragraph_context:
        parts.append(f"Additional context of the paragraph: {paragraph_context}")
    return "\n\n".join(parts), len(parts) > 2


# Converting answers


def _convert_concept_answers(
    concept: _Concept,
    position: int,
    answers: Mapping[str, _DecisionAnswer],
    decision_threshold: float = _DEFAULT_DECISION_THRESHOLD,
) -> tuple[_DecisionItemValue, _DecisionProbabilities] | None:
    """
    Converts the answers of a decision model to the questions of a concept
    (see :func:`_build_concept_questions`) to the value of the extracted item of
    the concept and to the probability distribution of the item.

    Value and probabilities by concept type:

    - Boolean concept: ``True`` if the probability ``p`` of "yes" is the decision
      threshold or more, else ``False``; ``{True: p, False: 1 - p}``.
    - Label concept with the "multi_class" classification type: a list with the most
      probable label; the probability of each label of the concept.
    - Label concept with the "multi_label" classification type: a list of the labels
      whose own predicate has a probability of the decision threshold or more, in
      the order of the labels of the concept; the probability of the predicate of each
      label of the concept (the probabilities do not sum to 1). No item if no label
      has a probability of the decision threshold or more.
    - Rating concept: the most probable integer of the rating scale; the probability
      of each integer of the scale.

    If several options are the most probable ones, the first of them is taken, in
    the order of the labels of the concept, or in the ascending order of the scale.
    The decision threshold does not apply to the options: the most probable one is
    taken, whatever its probability is.

    The probabilities of a "choice" or a "score" answer are taken as they are: they
    are not normalised, and an answer whose probabilities do not sum to 1 is valid.
    Such an answer must have the probability of each option offered, and of no
    other option: an option without a probability is not taken as an option with
    a zero probability. At least one of the probabilities must be above zero:
    an answer with a zero probability of each option tells nothing about
    the options, and is what an answer without the probabilities of the options
    offered becomes when the missing probabilities are filled in with zeros before
    the answer is converted. For the same reason, the option that a "choice" answer
    names as the picked one must be one of the options offered.

    Only the answers to the questions of the concept are read, i.e. the mapping may
    hold the answers to all the questions of a request.

    :param concept: Concept to convert the answers for. Only Boolean, label and
        rating concepts are supported.
    :type concept: _Concept
    :param position: Position of the concept in the sequence of concepts that
        the questions were built for.
    :type position: int
    :param answers: Answers of the decision model by the names of the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param decision_threshold: Lowest probability of "yes" with which a yes/no
        question of the concept is taken as answered "yes" (see
        :func:`_get_decision_threshold`). Defaults to 0.5.
    :type decision_threshold: float
    :return: Value of the extracted item (``bool`` for a Boolean concept, ``int`` for
        a rating concept, ``list[str]`` for a label concept) and its probability
        distribution, in the order of the options, or None if there is no item to
        extract, i.e. no label of a label concept with the "multi_label"
        classification type has a probability of the decision threshold or more.
    :rtype: tuple[bool | int | list[str], dict[bool | int | str, float]] | None
    :raises TypeError: If the concept is of a type that a decision model
        cannot extract.
    :raises _InvalidDecisionResponseError: If the answer to a question of the concept
        is missing, refused, of another kind than the question, without the needed
        probabilities, with a probability that is not a number within [0, 1],
        with options other than the options of the question, or with a zero
        probability of each option.
    """
    probabilities: _DecisionProbabilities

    if isinstance(concept, _BooleanConcept):
        probability = _get_predicate_probability(
            answers, _get_concept_question_name(position)
        )
        probabilities = {True: probability, False: 1 - probability}
        return probability >= decision_threshold, probabilities

    if isinstance(concept, _LabelConcept):
        if concept.classification_type == "multi_class":
            probabilities = _get_option_probabilities(
                answers,
                _get_concept_question_name(position),
                kind="choice",
                options=_get_label_options(concept),
            )
            # Safe cast: the options of the question are the labels of the concept
            return [cast(str, _get_most_probable_option(probabilities))], probabilities
        probabilities = {
            label: _get_predicate_probability(
                answers, _get_concept_question_name(position, label_position)
            )
            for label_position, label in enumerate(concept.labels)
        }
        labels = [
            label
            for label in concept.labels
            if probabilities[label] >= decision_threshold
        ]
        return (labels, probabilities) if labels else None

    if isinstance(concept, _RatingConcept):
        probabilities = _get_option_probabilities(
            answers,
            _get_concept_question_name(position),
            kind="score",
            options=_get_rating_options(concept),
        )
        # Safe cast: the levels of the question are the integers of the rating scale
        return cast(int, _get_most_probable_option(probabilities)), probabilities

    raise _get_unsupported_concept_error(concept)


def _convert_aspect_answer(
    position: int,
    answers: Mapping[str, _DecisionAnswer],
    decision_threshold: float = _DEFAULT_DECISION_THRESHOLD,
) -> tuple[bool, _DecisionProbabilities] | None:
    """
    Converts the answer of a decision model to the question of an aspect
    (see :func:`_build_aspect_questions`), asked on one paragraph, to whether
    the paragraph belongs to the aspect and to the probability distribution of that.

    The answer to the question of an aspect asked on one sentence is converted in
    the same way, to whether the sentence belongs to the aspect.

    A refused answer is not an error: it is not known whether the paragraph
    belongs to the aspect, i.e. the paragraph is not to be included in the aspect,
    while the answers on the other paragraphs are used. A model refuses, for
    example, to judge a text that has too little to judge, such as a bare heading.

    Only the answer to the question of the aspect is read, i.e. the mapping may hold
    the answers to the questions of all the aspects of a request.

    :param position: Position of the aspect in the sequence of aspects that
        the questions were built for.
    :type position: int
    :param answers: Answers of the decision model on the paragraph, by the names of
        the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param decision_threshold: Lowest probability of "yes" with which the paragraph
        is taken as belonging to the aspect (see :func:`_get_decision_threshold`).
        Defaults to 0.5.
    :type decision_threshold: float
    :return: Whether the paragraph belongs to the aspect, i.e. the probability ``p``
        of "yes" is the decision threshold or more, and ``{True: p, False: 1 - p}``,
        or None if the model refused to answer the question.
    :rtype: tuple[bool, dict[bool | int | str, float]] | None
    :raises _InvalidDecisionResponseError: If the answer to the question of the aspect
        is missing, of another kind than a predicate, without a probability, or
        with a probability that is not a number within [0, 1].
    """
    question_name = _get_aspect_question_name(position)
    if _is_refused_answer(answers, question_name):
        return None
    probability = _get_predicate_probability(answers, question_name)
    probabilities: _DecisionProbabilities = {True: probability, False: 1 - probability}
    return probability >= decision_threshold, probabilities


def _get_decision_threshold(
    instance: _Aspect | _Concept, model_decision_threshold: float
) -> float:
    """
    Returns the decision threshold that applies to an aspect or a concept: the one
    set for the aspect or the concept itself, if any, otherwise the one of
    the decision model that extracts it.

    Only the aspects and the concepts that are extracted with yes/no questions have
    a decision threshold of their own: aspects, Boolean concepts, and label
    concepts with the "multi_label" classification type.

    :param instance: Aspect or concept that is extracted.
    :type instance: _Aspect | _Concept
    :param model_decision_threshold: Decision threshold of the decision model.
    :type model_decision_threshold: float
    :return: Decision threshold that applies to the aspect or the concept.
    :rtype: float
    """
    decision_threshold = getattr(instance, "decision_threshold", None)
    return (
        model_decision_threshold if decision_threshold is None else decision_threshold
    )


def _validate_answers(
    questions: Sequence[_DecisionQuestion],
    answers: Mapping[str, _DecisionAnswer],
    accept_refusals: bool = False,
) -> None:
    """
    Validates that each of the questions has an answer that can be used, i.e.
    an answer that the conversion of the answers accepts (see
    :func:`_convert_concept_answers` and :func:`_convert_aspect_answer`).

    This allows for rejecting a response of a decision model as soon as it is
    received, e.g. before the other requests of the same extraction are sent,
    whatever the questions are built for.

    Only the answers to the given questions are read, i.e. the mapping may hold
    the answers to other questions as well.

    :param questions: Questions that were asked.
    :type questions: Sequence[_DecisionQuestion]
    :param answers: Answers of the decision model by the names of the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param accept_refusals: Whether a refused answer is accepted, as it is for
        the questions of aspects, whose conversion accepts it. Defaults to False,
        as for the questions of concepts.
    :type accept_refusals: bool
    :return: None
    :rtype: None
    :raises _InvalidDecisionResponseError: If the answer to a question is missing,
        refused while refusals are not accepted, of another kind than the question,
        without the needed probabilities, with a probability that is not a number
        within [0, 1], with options other than the options of the question, or
        with a zero probability of each option.
    """
    for question in questions:
        if accept_refusals and _is_refused_answer(answers, question.name):
            continue
        if question.kind == "predicate":
            _get_predicate_probability(answers, question.name)
        else:
            _get_option_probabilities(
                answers, question.name, kind=question.kind, options=question.options
            )


# Helpers


def _clean_text_for_question(text: str | None) -> str:
    """
    Cleans the text of a paragraph or of a sentence, or their additional context,
    for a text that the questions of aspects are asked on: as the text of
    a paragraph is cleaned when it is shown to an LLM for the extraction of
    aspects, i.e. without line breaks.

    :param text: Text to clean, or None, e.g. for additional context that
        is not set.
    :type text: str | None
    :return: The cleaned text, which is empty if there is nothing to show to
        a model.
    :rtype: str
    """
    return _clean_text_for_llm_prompt(
        text or "", preserve_linebreaks=False, strip_text=True
    )


def _get_concept_question_name(position: int, label_position: int | None = None) -> str:
    """
    Returns the name of a question of a concept.

    :param position: Position of the concept in the sequence of concepts.
    :type position: int
    :param label_position: Position of the label in the labels of the concept, for
        the question on one label. Defaults to None, i.e. the question is on
        the concept as a whole.
    :type label_position: int | None
    :return: Name of the question.
    :rtype: str
    """
    if label_position is None:
        return f"C{position}"
    return f"C{position}-L{label_position}"


def _get_aspect_question_name(position: int) -> str:
    """
    Returns the name of the question of an aspect.

    :param position: Position of the aspect in the sequence of aspects.
    :type position: int
    :return: Name of the question.
    :rtype: str
    """
    return f"A{position}"


def _get_label_options(concept: _LabelConcept) -> tuple[_DecisionOption, ...]:
    """
    Returns the options of the question of a label concept with the "multi_class"
    classification type: one option for each label, in the order of the labels.

    A label has no description, i.e. its text is the label of the option, and
    the option has no description.

    :param concept: Label concept.
    :type concept: _LabelConcept
    :return: Options of the question.
    :rtype: tuple[_DecisionOption, ...]
    """
    return tuple(_DecisionOption(value=label, label=label) for label in concept.labels)


def _get_rating_options(concept: _RatingConcept) -> tuple[_DecisionOption, ...]:
    """
    Returns the levels of the question of a rating concept: one level for each
    integer of the rating scale, in ascending order.

    :param concept: Rating concept.
    :type concept: _RatingConcept
    :return: Levels of the question.
    :rtype: tuple[_DecisionOption, ...]
    """
    start, end = concept._rating_start, concept._rating_end
    return tuple(
        _DecisionOption(
            value=level,
            label=str(level),
            description=f"{level} on a scale from {start} to {end}",
        )
        for level in range(start, end + 1)
    )


@cache
def _get_instructions_template(template_name: str) -> Template:
    """
    Returns the template of the question instructions. The template is loaded once.

    :param template_name: Name of the template, i.e. the name of its file without
        the extension.
    :type template_name: str
    :return: Template of the question instructions.
    :rtype: Template
    """
    return _get_template(template_name)


def _render_instructions(template_name: str, **params: Any) -> str:
    """
    Renders the instructions of a question.

    :param template_name: Name of the template of the instructions.
    :type template_name: str
    :param params: Params of the template.
    :type params: Any
    :return: Instructions of the question.
    :rtype: str
    """
    return _get_instructions_template(template_name).render(**params).strip()


def _render_concept_question_instructions(
    concept: _Concept,
    question_type: Literal["boolean", "multi_class", "multi_label", "rating"],
    **params: Any,
) -> str:
    """
    Renders the instructions of a question of a concept.

    :param concept: Concept that the question is built for.
    :type concept: _Concept
    :param question_type: Type of the question, which selects the wording of
        the instructions.
    :type question_type: Literal["boolean", "multi_class", "multi_label", "rating"]
    :param params: Params of the template that are specific to the type of
        the question: the label, or the start and the end of the rating scale.
    :type params: Any
    :return: Instructions of the question.
    :rtype: str
    """
    return _render_instructions(
        _CONCEPT_QUESTION_TEMPLATE_NAME,
        question_type=question_type,
        concept_name=concept.name,
        concept_description=concept.description,
        **params,
    )


def _get_unsupported_concept_error(concept: object) -> TypeError:
    """
    Returns the error for a concept of a type that a decision model cannot extract.

    :param concept: Concept of the unsupported type.
    :type concept: object
    :return: Error that names the concept.
    :rtype: TypeError
    """
    return TypeError(
        f"Concept `{getattr(concept, 'name', concept)}` of type "
        f"`{type(concept).__name__}` cannot be extracted by a decision model: only "
        f"Boolean, label and rating concepts can be."
    )


def _is_probability(value: object) -> bool:
    """
    Checks whether a value is a probability, i.e. a number within [0, 1].

    :param value: Value to check.
    :type value: object
    :return: True if the value is a number within [0, 1], False otherwise. A boolean
        and a NaN are not probabilities.
    :rtype: bool
    """
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0 <= value <= 1
    )


def _is_refused_answer(
    answers: Mapping[str, _DecisionAnswer], question_name: str
) -> bool:
    """
    Checks whether the model refused to answer a question.

    :param answers: Answers of the decision model by the names of the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param question_name: Name of the question.
    :type question_name: str
    :return: True if the question has a refused answer, False otherwise, also if
        the question has no answer.
    :rtype: bool
    """
    answer = answers.get(question_name)
    return answer is not None and answer.refused


def _get_answer(
    answers: Mapping[str, _DecisionAnswer],
    question_name: str,
    kind: _DecisionKind,
) -> _DecisionAnswer:
    """
    Returns the answer to a question, checked for being an answer of the model to
    a question of that kind.

    :param answers: Answers of the decision model by the names of the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param question_name: Name of the question.
    :type question_name: str
    :param kind: Kind of the question.
    :type kind: Literal["predicate", "choice", "score"]
    :return: Answer to the question.
    :rtype: _DecisionAnswer
    :raises _InvalidDecisionResponseError: If the answer is missing, refused, or
        of another kind than the question.
    """
    answer = answers.get(question_name)
    if answer is None:
        raise _InvalidDecisionResponseError(question_name, "the answer is missing.")
    if answer.refused:
        raise _InvalidDecisionResponseError(
            question_name, "the model refused to answer."
        )
    if answer.kind != kind:
        raise _InvalidDecisionResponseError(
            question_name,
            f"the answer is of the kind `{answer.kind}`, while the question is of "
            f"the kind `{kind}`.",
        )
    return answer


def _get_predicate_probability(
    answers: Mapping[str, _DecisionAnswer], question_name: str
) -> float:
    """
    Returns the probability of "yes" from the answer to a predicate.

    :param answers: Answers of the decision model by the names of the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param question_name: Name of the predicate.
    :type question_name: str
    :return: Probability of "yes".
    :rtype: float
    :raises _InvalidDecisionResponseError: If the answer is missing, refused, of
        another kind than a predicate, without a probability, or with a probability
        that is not a number within [0, 1].
    """
    answer = _get_answer(answers, question_name, kind="predicate")
    if answer.probability is None:
        raise _InvalidDecisionResponseError(
            question_name, "the answer has no probability."
        )
    if not _is_probability(answer.probability):
        raise _InvalidDecisionResponseError(
            question_name,
            f"the probability {answer.probability!r} is not a number within [0, 1].",
        )
    return float(answer.probability)


def _get_option_probabilities(
    answers: Mapping[str, _DecisionAnswer],
    question_name: str,
    kind: Literal["choice", "score"],
    options: tuple[_DecisionOption, ...],
) -> _DecisionProbabilities:
    """
    Returns the probabilities of the options from the answer to a "choice" or
    a "score" question.

    The probabilities are returned as they are, i.e. they are not normalised, and
    they do not have to sum to 1. At least one of them must be above zero.

    :param answers: Answers of the decision model by the names of the questions.
    :type answers: Mapping[str, _DecisionAnswer]
    :param question_name: Name of the question.
    :type question_name: str
    :param kind: Kind of the question.
    :type kind: Literal["choice", "score"]
    :param options: Options of the question.
    :type options: tuple[_DecisionOption, ...]
    :return: Probability of each option of the question, by the value of the option
        (see :attr:`_DecisionOption.value`), in the order of the options.
    :rtype: dict[bool | int | str, float]
    :raises _InvalidDecisionResponseError: If the answer is missing, refused, of
        another kind than the question, without the probabilities of the options,
        with an option that is not an option of the question, without an option of
        the question, with a probability that is not a number within [0, 1], with
        a zero probability of each option, or with an option named as the picked
        one that is not an option of the question.
    """
    answer = _get_answer(answers, question_name, kind=kind)
    option_probabilities = answer.option_probabilities
    if not option_probabilities:
        raise _InvalidDecisionResponseError(
            question_name, "the answer has no probabilities of the options."
        )
    option_labels = [option.label for option in options]
    not_offered_labels = [
        label for label in option_probabilities if label not in option_labels
    ]
    if not_offered_labels:
        raise _InvalidDecisionResponseError(
            question_name,
            f"the answer has options that were not offered: {not_offered_labels}. "
            f"Options offered: {option_labels}.",
        )
    missing_labels = [
        label for label in option_labels if label not in option_probabilities
    ]
    if missing_labels:
        raise _InvalidDecisionResponseError(
            question_name,
            f"the answer has no probabilities of the options {missing_labels}.",
        )
    probabilities: _DecisionProbabilities = {}
    for option in options:
        probability = option_probabilities[option.label]
        if not _is_probability(probability):
            raise _InvalidDecisionResponseError(
                question_name,
                f"the probability {probability!r} of the option `{option.label}` "
                f"is not a number within [0, 1].",
            )
        probabilities[option.value] = float(probability)
    if answer.chosen_option is not None and answer.chosen_option not in option_labels:
        raise _InvalidDecisionResponseError(
            question_name,
            f"the answer picks the option `{answer.chosen_option}`, which was not "
            f"offered. Options offered: {option_labels}.",
        )
    if not any(probabilities.values()):
        # The answer does not tell the options apart. This is also what an answer
        # without the probabilities of the options offered looks like once
        # the missing probabilities are filled in with zeros, as the library
        # that sends the requests does.
        raise _InvalidDecisionResponseError(
            question_name,
            "the probability of each option is zero, i.e. the answer has no usable "
            "probabilities of the options offered.",
        )
    return probabilities


def _get_most_probable_option(
    probabilities: _DecisionProbabilities,
) -> bool | int | str:
    """
    Returns the most probable option. If several options are the most probable
    ones, the first of them is returned.

    :param probabilities: Probability of each option, in the order of the options.
    :type probabilities: dict[bool | int | str, float]
    :return: The most probable option.
    :rtype: bool | int | str
    """
    # `max` returns the first of the options with the highest probability
    return max(probabilities, key=probabilities.__getitem__)
