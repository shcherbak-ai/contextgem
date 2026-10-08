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

"""
Module defining tests for usage examples.
"""

from __future__ import annotations

import pytest

from tests.base import BaseTest


class TestUsageExamples(BaseTest):
    """
    Tests for the usage examples in the documentation, README and docstrings.
    """

    @pytest.mark.vcr
    def test_usage_examples(self):
        """
        Tests for usage examples in project's documentation and README.md.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        """
        from dev.usage_examples.docs.advanced import (
            advanced_aspects_and_concepts_document,  # noqa: F401
            advanced_aspects_with_concepts,  # noqa: F401
            advanced_multiple_docs_pipeline,  # noqa: F401
        )
        from dev.usage_examples.docs.aspects import (
            aspect_with_concepts,  # noqa: F401
            aspect_with_justifications,  # noqa: F401
            aspect_with_sub_aspects,  # noqa: F401
            basic_aspect,  # noqa: F401
            complex_hierarchy,  # noqa: F401
        )
        from dev.usage_examples.docs.concepts.boolean_concept import (
            boolean_concept,  # noqa: F401
            refs_and_justifications,
        )
        from dev.usage_examples.docs.concepts.date_concept import (
            date_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.json_object_concept import (
            adding_examples,  # noqa: F401
            json_object_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.json_object_concept.structure import (
            nested_class_structure,  # noqa: F401
            nested_structure,  # noqa: F401
            simple_class_structure,  # noqa: F401
            simple_structure,  # noqa: F401
        )
        from dev.usage_examples.docs.concepts.label_concept import (
            document_aspect_analysis,  # noqa: F401
            label_concept,  # noqa: F401
            multi_label_classification,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.numerical_concept import (
            numerical_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.rating_concept import (
            multiple_ratings,  # noqa: F401
            rating_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.string_concept import (
            adding_examples,  # noqa: F401,F811
            refs_and_justifications,  # noqa: F401,F811
            string_concept,  # noqa: F401
        )
        from dev.usage_examples.docs.llm_config import (
            chat_with_tools,  # noqa: F401
            cost_tracking,  # noqa: F401
            detailed_usage,  # noqa: F401
            fallback_models,  # noqa: F401
            llm_api,  # noqa: F401
            llm_local,  # noqa: F401
            model_group,  # noqa: F401
            reasoning_params,  # noqa: F401
            tracking_usage_and_cost,  # noqa: F401
        )
        from dev.usage_examples.docs.llms.llm_extraction_methods import (
            extract_all,  # noqa: F401
            extract_aspects_from_document,  # noqa: F401
            extract_concepts_from_aspect,  # noqa: F401
            extract_concepts_from_document,  # noqa: F401
        )
        from dev.usage_examples.docs.llms.llm_init import (
            llm_api,  # noqa: F401,F811
            llm_local,  # noqa: F401,F811
            lm_studio_connection_error_fix,  # noqa: F401
        )
        from dev.usage_examples.docs.optimizations import (
            optimization_accuracy,  # noqa: F401
            optimization_choosing_llm,  # noqa: F401
            optimization_cost,  # noqa: F401
            optimization_long_docs,  # noqa: F401
            optimization_speed,  # noqa: F401
        )
        from dev.usage_examples.docs.quickstart import (
            quickstart_aspect,  # noqa: F401
            quickstart_concept_aspect,  # noqa: F401
            quickstart_concept_document_text,  # noqa: F401
            quickstart_concept_document_vision,  # noqa: F401
            quickstart_sub_aspect,  # noqa: F401
        )
        from dev.usage_examples.docs.serialization import serialization  # noqa: F401
        from dev.usage_examples.readme import (
            llm_chat,  # noqa: F401
            quickstart_aspect,  # noqa: F401,F811
            quickstart_concept,  # noqa: F401
        )

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
