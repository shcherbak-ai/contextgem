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
Module defining tests for memory usage and extraction benchmarks.
"""

from __future__ import annotations

import pytest

from contextgem.internal.loggers import logger
from tests.base import BaseTest
from tests.benchmark.core import run_benchmark_for_module
from tests.memory_profiling import check_locals_memory_usage, memory_profile_and_capture


class TestPerformance(BaseTest):
    """
    Tests that establish the memory usage baseline of the test run
    and execute the extraction benchmarks.
    """

    @memory_profile_and_capture
    def test_establish_memory_baseline(self):
        """
        Establishes a memory usage baseline for subsequent per-method memory calculations.

        This test captures the initial memory footprint including module imports,
        variable initialization, and framework setup. The memory profile from this
        test serves as a baseline against which memory deltas of other test methods
        can be measured.
        """
        logger.info("Memory usage baseline captured.")

    @pytest.mark.vcr
    @memory_profile_and_capture
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

        run_benchmark_for_module(
            llm=self.llm_extractor,
            judge_llm=self.llm_reasoner,
            module_name="tests.benchmark.configs.dev_contract",
            benchmark_name="Dev Contract (Text Only)",
        )

        check_locals_memory_usage(locals(), test_name="test_benchmarks")
