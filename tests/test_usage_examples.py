# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for usage examples.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from tests.base import BaseTest


_USAGE_EXAMPLES_DIR = Path(__file__).parent.parent / "dev" / "usage_examples"

# The usage examples of the documentation and of the README that send requests to
# the API of a model, as the names of their modules in `dev/usage_examples/`. Each
# of them is run by a test of its own, with a cassette of its own, so that a change
# in an example requires re-recording the cassette of that example only.
_EXAMPLES_WITH_REQUESTS = [
    "docs.advanced.advanced_aspects_and_concepts_document",
    "docs.advanced.advanced_aspects_with_concepts",
    "docs.advanced.advanced_multiple_docs_pipeline",
    "docs.aspects.aspect_with_concepts",
    "docs.aspects.aspect_with_justifications",
    "docs.aspects.aspect_with_sub_aspects",
    "docs.aspects.basic_aspect",
    "docs.aspects.complex_hierarchy",
    "docs.concepts.boolean_concept.boolean_concept",
    "docs.concepts.boolean_concept.refs_and_justifications",
    "docs.concepts.date_concept.date_concept",
    "docs.concepts.date_concept.refs_and_justifications",
    "docs.concepts.json_object_concept.adding_examples",
    "docs.concepts.json_object_concept.json_object_concept",
    "docs.concepts.json_object_concept.refs_and_justifications",
    "docs.concepts.label_concept.document_aspect_analysis",
    "docs.concepts.label_concept.label_concept",
    "docs.concepts.label_concept.multi_label_classification",
    "docs.concepts.label_concept.refs_and_justifications",
    "docs.concepts.numerical_concept.numerical_concept",
    "docs.concepts.numerical_concept.refs_and_justifications",
    "docs.concepts.rating_concept.multiple_ratings",
    "docs.concepts.rating_concept.rating_concept",
    "docs.concepts.rating_concept.refs_and_justifications",
    "docs.concepts.string_concept.adding_examples",
    "docs.concepts.string_concept.refs_and_justifications",
    "docs.concepts.string_concept.string_concept",
    "docs.decision_models.decision_model_aspects",
    "docs.decision_models.decision_model_concepts",
    "docs.decision_models.decision_model_group",
    "docs.decision_models.decision_model_usage_and_cost",
    "docs.llm_config.chat_with_tools",
    "docs.llm_config.chat_with_tools_simple",
    "docs.llms.llm_extraction_methods.extract_all",
    "docs.llms.llm_extraction_methods.extract_aspects_from_document",
    "docs.llms.llm_extraction_methods.extract_concepts_from_aspect",
    "docs.llms.llm_extraction_methods.extract_concepts_from_document",
    "docs.optimizations.optimization_accuracy",
    "docs.quickstart.quickstart_aspect",
    "docs.quickstart.quickstart_concept_aspect",
    "docs.quickstart.quickstart_concept_document_text",
    "docs.quickstart.quickstart_concept_document_vision",
    "docs.quickstart.quickstart_sub_aspect",
    "docs.serialization.serialization",
    "readme.llm_chat",
    "readme.quickstart_decision_model",
    "readme.quickstart_llm",
]

# The usage examples that send no requests, e.g. the ones that only create objects.
# They are run without a cassette, i.e. with all network access blocked.
_EXAMPLES_WITHOUT_REQUESTS = [
    "docs.concepts.json_object_concept.structure.nested_class_structure",
    "docs.concepts.json_object_concept.structure.nested_structure",
    "docs.concepts.json_object_concept.structure.simple_class_structure",
    "docs.concepts.json_object_concept.structure.simple_structure",
    "docs.decision_models.decision_model_fallbacks",
    "docs.decision_models.decision_model_init",
    "docs.decision_models.decision_model_local",
    "docs.decision_models.decision_model_threshold",
    "docs.llm_config.cost_tracking",
    "docs.llm_config.detailed_usage",
    "docs.llm_config.fallback_models",
    "docs.llm_config.model_group",
    "docs.llm_config.reasoning_params",
    "docs.llm_config.tracking_usage_and_cost",
    "docs.llms.llm_init.llm_api",
    "docs.llms.llm_init.llm_local",
    "docs.optimizations.optimization_choosing_llm",
    "docs.optimizations.optimization_cost",
    "docs.optimizations.optimization_long_docs",
    "docs.optimizations.optimization_speed",
]


def _run_usage_example(example: str) -> None:
    """
    Runs a usage example by importing its module.

    A module is executed only when it is first imported. If the module is already
    imported, e.g. by another test, it is removed from the imported modules first,
    so that the example is run by the test that calls this function.

    :param example: The name of the module of the example in `dev/usage_examples/`,
        e.g. "docs.quickstart.quickstart_aspect".
    :type example: str
    :return: None
    :rtype: None
    """
    module_name = f"dev.usage_examples.{example}"
    sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


class TestUsageExamples(BaseTest):
    """
    Tests for the usage examples in the documentation, README and docstrings.
    """

    @pytest.mark.vcr
    @pytest.mark.parametrize("example", _EXAMPLES_WITH_REQUESTS)
    def test_usage_example(self, example: str):
        """
        Tests a usage example of the documentation or of the README that sends
        requests to the API of a model. The cassette of the test has the requests
        of that example only.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        """
        _run_usage_example(example)

    @pytest.mark.parametrize("example", _EXAMPLES_WITHOUT_REQUESTS)
    def test_usage_example_without_requests(self, example: str):
        """
        Tests a usage example of the documentation that sends no requests.
        """
        _run_usage_example(example)

    def test_usage_examples_are_listed(self):
        """
        Tests that each usage example of the documentation and of the README is run
        by one of the tests above, i.e. that an example that is added is also listed.
        """
        found = {
            ".".join(file.relative_to(_USAGE_EXAMPLES_DIR).with_suffix("").parts)
            for directory in ("docs", "readme")
            for file in (_USAGE_EXAMPLES_DIR / directory).rglob("*.py")
            if file.name != "__init__.py"
        }
        with_requests = set(_EXAMPLES_WITH_REQUESTS)
        without_requests = set(_EXAMPLES_WITHOUT_REQUESTS)
        assert len(with_requests) == len(_EXAMPLES_WITH_REQUESTS)
        assert len(without_requests) == len(_EXAMPLES_WITHOUT_REQUESTS)
        assert not with_requests & without_requests
        assert found == with_requests | without_requests

    @pytest.mark.usefixtures("restore_logger_settings")
    def test_docstring_examples(self):
        """
        Tests for examples in docstrings.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        Note: One of the examples disables the logger, so the logger settings are restored after the test.
        """
        from dev.usage_examples.docstrings.aspects import def_aspect  # noqa: F401
        from dev.usage_examples.docstrings.concepts import (
            def_boolean_concept,  # noqa: F401
            def_date_concept,  # noqa: F401
            def_json_object_concept,  # noqa: F401
            def_label_concept,  # noqa: F401
            def_numerical_concept,  # noqa: F401
            def_rating_concept,  # noqa: F401
            def_string_concept,  # noqa: F401
        )
        from dev.usage_examples.docstrings.data_models import (
            def_model_pricing,  # noqa: F401
        )
        from dev.usage_examples.docstrings.decision_models import (
            def_decision_model,  # noqa: F401
        )
        from dev.usage_examples.docstrings.documents import def_document  # noqa: F401
        from dev.usage_examples.docstrings.examples import (
            def_example_json_object,  # noqa: F401
            def_example_string,  # noqa: F401
        )
        from dev.usage_examples.docstrings.images import def_image  # noqa: F401
        from dev.usage_examples.docstrings.llms import (  # noqa: F401
            def_llm,
            def_model_group,
        )
        from dev.usage_examples.docstrings.paragraphs import def_paragraph  # noqa: F401
        from dev.usage_examples.docstrings.pipelines import def_pipeline  # noqa: F401
        from dev.usage_examples.docstrings.sentences import def_sentence  # noqa: F401
        from dev.usage_examples.docstrings.utils import (
            json_object_cls_struct,  # noqa: F401
            reload_logger_settings,  # noqa: F401
        )
