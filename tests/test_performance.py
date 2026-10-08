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
Module defining tests for extraction benchmarks.
"""

from __future__ import annotations

import pytest

from contextgem.public import DocumentLLM
from tests.base import BaseTest
from tests.benchmark.core import run_benchmark_for_module


class TestPerformance(BaseTest):
    """
    Tests that execute the extraction benchmarks.
    """

    @pytest.mark.vcr
    def test_benchmarks(self):
        """
        Executes extraction benchmarks to evaluate the accuracy and reliability
        of the extraction pipeline.

        This test runs a suite of benchmark scenarios and asserts that the total benchmark
        scores meet or exceed a set minimum score.

        The benchmarks are particularly important for validating the impact of prompt changes
        and ensuring that extraction quality remains high as the system evolves. If a benchmark
        score falls below the threshold, the test fails, signaling a potential regression in
        extraction performance.
        """

        # The benchmark unsets the system message of the judge LLM, so the judge
        # is a new LLM with the settings of the reasoner, not the shared reasoner
        judge_llm = DocumentLLM(**self._llm_reasoner_kwargs_openai)  # ty: ignore[invalid-argument-type]

        run_benchmark_for_module(
            llm=self.llm_extractor,
            judge_llm=judge_llm,
            module_name="tests.benchmark.configs.dev_contract",
            benchmark_name="Dev Contract (Text Only)",
        )
