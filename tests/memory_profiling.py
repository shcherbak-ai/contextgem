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
Memory profiling utilities for ContextGem testing.

This module provides tools for measuring and analyzing memory usage of ContextGem
during test execution. It includes functions for object memory measurement, test-wide
memory footprint analysis, and decorators for automated memory profiling.

Memory profiling is optional and can be controlled via configuration. When disabled,
all profiling functions return early without performing measurements, allowing tests
to run without the overhead of memory tracking.

All memory measurements are reported in MiB (Mebibytes, 1 MiB = 1,048,576 bytes)
for consistency across pympler object-level and memory_profiler process-level measurements.

Key features:
- Object-level memory usage measurement with pympler (MiB)
- Test function memory footprint analysis with memory_profiler (MiB)
- Automated memory profiling with decorators
- Memory usage data extraction and reporting
- Optional profiling that can be enabled/disabled via command line flag
"""

from __future__ import annotations

import io
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from memory_profiler import profile
from pympler import asizeof

from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.base.llms import _GenericLLMProcessor
from contextgem.internal.loggers import logger
from tests.conftest import is_memory_profiling_enabled


# Default memory limits (in MiB) for memory usage checks.
# These should be reasonable limits to cover most comprehensive and complex test cases,
# as well as heavy objects such as images.
MAX_OBJECT_MEMORY = 1.0  # Maximum memory per object in MiB
MAX_MEMORY_ALL_OBJECTS = 10.0  # Maximum total memory for all objects in MiB
MAX_TOTAL_MEMORY_PER_TEST = (
    100.0  # Maximum memory footprint delta per test method in MiB
)
# Maximum baseline memory footprint in MiB.
MAX_BASELINE_MEMORY = 1000.0

DEFAULT_TARGET_CLASSES_FOR_MEMORY_USAGE_PROFILING = (
    _InstanceBase,
    _GenericLLMProcessor,
)

F = TypeVar("F", bound=Callable[..., Any])


def check_object_memory_usage(
    obj: Any,
    obj_name: str = "Object",
    deep: bool = False,
    max_obj_memory: float = MAX_OBJECT_MEMORY,
) -> float:
    """
    Measures and logs the memory usage of a Python object using pympler.
    Raises an AssertionError if memory usage exceeds the specified limit.

    :param obj: The Python object whose memory usage is to be measured.
    :param obj_name: A descriptive name for the object (used in logging).
    :type obj_name: str
    :param deep: Whether to perform deep memory calculation including referenced objects.
        If False, only calculates the flat size of the object itself.
    :type deep: bool
    :param max_obj_memory: Maximum allowed memory usage in MiB. Defaults to MAX_OBJECT_MEMORY.
    :type max_obj_memory: float
    :return: Memory usage in MiB.
    :rtype: float
    :raises AssertionError: If actual memory usage exceeds max_obj_memory.

    Note: This function uses pympler.asizeof for accurate memory measurement:
    - Deep calculation: Uses asizeof() which recursively includes all referenced objects
    - Shallow calculation: Uses flatsize() which calculates basic size + item size x length
    """

    # Check if memory profiling is disabled
    if not is_memory_profiling_enabled():
        logger.debug(
            f"Memory profiling disabled - skipping memory check for {obj_name}"
        )
        return 0.0

    if deep:
        # Use pympler's asizeof for deep memory calculation including all references
        size_bytes = asizeof.asizeof(obj)
        calculation_type = "deep"
    else:
        # Use pympler's flatsize for shallow calculation (basic + item sizes)
        size_bytes = asizeof.flatsize(obj)
        calculation_type = "shallow"

    size_mib = cast(int, size_bytes) / (1024 * 1024)  # MiB

    logger.debug(
        f"Memory usage of {obj_name} ({calculation_type}): "
        f"{size_bytes:,} bytes ({size_mib:.6f} MiB)"
    )

    # Check if memory usage exceeds the specified limit
    if size_mib > max_obj_memory:
        raise AssertionError(
            f"Memory usage of {obj_name} ({size_mib:.6f} MiB) exceeds the maximum "
            f"allowed limit of {max_obj_memory:.6f} MiB"
        )

    return size_mib


def check_locals_memory_usage(
    test_locals: dict[str, Any],
    test_name: str = "test",
    target_classes: tuple[
        type, ...
    ] = DEFAULT_TARGET_CLASSES_FOR_MEMORY_USAGE_PROFILING,
    max_obj_memory: float = MAX_OBJECT_MEMORY,
    max_memory_all_objects: float = MAX_MEMORY_ALL_OBJECTS,
) -> float:
    """
    Measure and report the memory footprint of target objects from test function locals.
    Raises AssertionError if memory usage exceeds specified limits.

    By default, this function tracks ContextGem objects
    (DEFAULT_TARGET_CLASSES_FOR_MEMORY_USAGE_PROFILING), but can be customized to track
    any specified classes by passing target_classes.

    This function should be called at the end of test functions with `locals()` as parameter
    to measure the memory footprint of target objects created during the test.

    Important limitations:
    - Only captures objects that are assigned to variables (i.e., present in locals())
    - Only captures the final/last state of such objects at the time of the function call
    - Does not capture temporary objects created without variable assignment
    - Does not capture intermediate states of modified objects

    Usage:
        def test_my_function():
            # ... test code ...

            # At the end of the test (uses default limits):
            check_locals_memory_usage(locals(), "test_my_function")

        # Or with custom classes to track:
        def test_with_custom_classes():
            # ... test code ...

            check_locals_memory_usage(
                locals(),
                "test_with_custom_classes",
                target_classes=(MyCustomClass, AnotherClass)
            )

        # With custom memory limits:
        def test_with_custom_memory_limits():
            # ... test code ...

            check_locals_memory_usage(
                locals(),
                "test_with_custom_memory_limits",
                max_obj_memory=1.0,  # Each object max 1 MiB
                max_memory_all_objects=10.0,  # Total max 10 MiB
            )

    :param test_locals: The `locals()` dictionary from the test function
    :type test_locals: dict[str, Any]
    :param test_name: Name of the test for logging purposes
    :type test_name: str
    :param target_classes: Classes to check for in object MRO. Defaults to
        ContextGem classes specified in
        DEFAULT_TARGET_CLASSES_FOR_MEMORY_USAGE_PROFILING
    :type target_classes: tuple[type, ...]
    :param max_obj_memory: Maximum allowed memory usage per object in MiB.
        Defaults to MAX_OBJECT_MEMORY.
    :type max_obj_memory: float
    :param max_memory_all_objects: Maximum allowed total memory usage of
        all objects in MiB. Defaults to MAX_MEMORY_ALL_OBJECTS.
    :type max_memory_all_objects: float
    :return: Total objects' memory usage in MiB
    :rtype: float
    :raises AssertionError: If max_obj_memory is exceeded by any object or if
        max_memory_all_objects is exceeded by the total memory usage of all objects.
    """

    # Check if memory profiling is disabled
    if not is_memory_profiling_enabled():
        logger.debug(
            f"Memory profiling disabled - skipping locals memory check for {test_name}"
        )
        return 0.0

    def is_target_object(obj: Any) -> bool:
        """
        True when obj inherits from any of the target classes.

        :param obj: The object to check if it inherits from any of the target classes.
        :type obj: Any
        :return: True if obj inherits from any of the target classes, False otherwise.
        :rtype: bool
        """
        try:
            # Handle both instances and class objects
            if isinstance(obj, type):
                # obj is a class, check if it's a subclass of any target class
                return any(
                    issubclass(obj, target_class) for target_class in target_classes
                )
            else:
                # obj is an instance, check its MRO for any target class
                mro = obj.__class__.mro()
                return any(target_class in mro for target_class in target_classes)
        except (AttributeError, TypeError):
            # Skip objects that cause issues during type checking
            logger.warning(
                f"Skipping object {obj} from memory usage analysis due to target class error"
            )
            return False

    # Find target objects in the test's locals
    captured_objects: list[Any] = []

    logger.debug(f"Analyzing {len(test_locals)} local variables in {test_name}")

    for var_name, var_value in test_locals.items():
        if is_target_object(var_value):
            captured_objects.append(var_value)
            logger.debug(
                f"Found target object: {var_name} = {var_value.__class__.__name__}"
            )
        # Also check if it's a list/tuple containing target objects
        elif isinstance(var_value, list | tuple):
            for i, item in enumerate(var_value):
                if is_target_object(item):
                    captured_objects.append(item)
                    logger.debug(
                        f"Found target object in {var_name}[{i}]: {item.__class__.__name__}"
                    )
        # Check if it's a dict containing target objects
        elif isinstance(var_value, dict):
            for key, item in var_value.items():
                if is_target_object(item):
                    captured_objects.append(item)
                    logger.debug(
                        f"Found target object in {var_name}[{key}]: {item.__class__.__name__}"
                    )

    # Remove duplicates while preserving order
    seen_ids = set()
    unique_objects = []
    for obj in captured_objects:
        obj_id = id(obj)
        if obj_id not in seen_ids:
            seen_ids.add(obj_id)
            unique_objects.append(obj)

    logger.debug(f"Found {len(unique_objects)} unique target objects in {test_name}")

    if not unique_objects:
        logger.debug(f"No target objects found in {test_name}")
        return 0.0

    # Group objects by class for better reporting
    objects_by_class = defaultdict(list)
    for obj in unique_objects:
        objects_by_class[obj.__class__.__name__].append(obj)

    # Calculate memory usage per class
    total_size = 0
    class_reports = []

    for class_name, objects in objects_by_class.items():
        class_total = 0
        for obj in objects:
            # Use the existing memory check function with deep calculation
            size = check_object_memory_usage(
                obj,
                f"{class_name}_{id(obj)}",
                deep=True,
                max_obj_memory=max_obj_memory,
            )
            class_total += size

        class_reports.append(
            {
                "class_name": class_name,
                "count": len(objects),
                "size_mib": class_total,
            }
        )
        total_size += class_total

    # Report summary
    logger.debug(
        f"*** {test_name}: analyzed {len(unique_objects)} target objects "
        f"using {total_size:.6f} MiB total"
    )

    # Report details by class (sorted by memory usage, largest first)
    for report in sorted(class_reports, key=lambda x: x["size_mib"], reverse=True):
        logger.debug(
            f"  • {report['class_name']}: {report['count']} objects, "
            f"{report['size_mib']:.6f} MiB"
        )

    # Check if total memory usage exceeds the specified limit
    if total_size > max_memory_all_objects:
        raise AssertionError(
            f"Total memory usage in {test_name} ({total_size:.6f} MiB) exceeds the maximum "
            f"allowed limit of {max_memory_all_objects:.6f} MiB"
        )

    return total_size


def memory_profile_and_capture(
    func: F | None = None, *, max_memory: float = MAX_TOTAL_MEMORY_PER_TEST
) -> F | Callable[[F], F]:
    """
    Decorator that wraps a test method with memory_profiler and captures its output.
    Stores the profiling output in the class's `memory_profiles` dictionary.

    Automatically runs test_establish_memory_baseline if it hasn't been executed yet
    to capture the baseline memory footprint. Calculates memory delta for each test
    method relative to baseline and all previously executed tests, then validates
    against max_memory limit.

    Usage:
        # Without parameters (uses default max_memory):
        @memory_profile_and_capture
        def test_method(self):
            # test code here
            pass

        # With custom max_memory:
        @memory_profile_and_capture(max_memory=1.0)
        def test_method(self):
            # test code here
            pass

    The profiling output will be stored in `self.memory_profiles[method_name]`.

    Requires that the class has a `memory_profiles` attribute (dict) either as an
    instance attribute or class attribute. Raises AttributeError if not found.

    :param func: The function to be profiled (None when used with parameters)
    :type func: F | None
    :param max_memory: Maximum allowed memory delta in MiB. Defaults to
        MAX_TOTAL_MEMORY_PER_TEST.
    :type max_memory: float
    :return: The wrapped function or decorator
    :rtype: F | Callable[[F], F]
    :raises AttributeError: If neither `self.memory_profiles` nor
        `self.__class__.memory_profiles` exists
    :raises AssertionError: If the method's memory delta exceeds max_memory
    """

    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            # Check if memory profiling is disabled
            if not is_memory_profiling_enabled():
                logger.debug(
                    f"Memory profiling disabled - running {f.__name__} without profiling"
                )
                return f(self, *args, **kwargs)

            # Helper function to get or set attribute in instance or class
            def get_or_set_memory_attribute(
                attr_name: str, value: Any | None = None
            ) -> dict | None:
                if hasattr(self, attr_name):
                    if value:
                        setattr(self, attr_name, value)
                    return getattr(self, attr_name)
                elif hasattr(self.__class__, attr_name):
                    if value:
                        setattr(self.__class__, attr_name, value)
                    return getattr(self.__class__, attr_name)
                else:
                    raise AttributeError(
                        f"Neither instance nor class has `{attr_name}` attribute. "
                        f"Please add `{attr_name}` dict attribute to the class "
                        f"definition before using `@memory_profile_and_capture` decorator."
                    )

            # Get the memory containers
            memory_profiles = get_or_set_memory_attribute("memory_profiles")
            memory_deltas = get_or_set_memory_attribute("memory_deltas")

            # Check if baseline has been established
            baseline_method_name = "test_establish_memory_baseline"
            if (
                # Safe cast: memory_profiles is a dict
                baseline_method_name not in cast(dict, memory_profiles)
                and f.__name__ != baseline_method_name
                and hasattr(self, baseline_method_name)
            ):
                # Run the baseline method first
                logger.debug("Running baseline memory profiling first...")
                baseline_method = getattr(self, baseline_method_name)
                # Run baseline method with profiling
                baseline_buf = io.StringIO()
                baseline_original_stdout = sys.stdout
                sys.stdout = baseline_buf
                try:
                    baseline_profiled_func = profile(baseline_method)
                    baseline_profiled_func()  # type: ignore
                finally:
                    sys.stdout = baseline_original_stdout
                    # Store the captured profiling output in the class's memory_profiles dictionary
                    baseline_profile_output = baseline_buf.getvalue()
                    # Safe cast: memory_profiles is a dict
                    cast(dict, memory_profiles)[baseline_method_name] = (
                        baseline_profile_output
                    )
                    logger.debug("Baseline memory profiling completed.")

            # Create a StringIO buffer to capture stdout
            buf = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = buf

            try:
                # Apply memory profiler to the method
                profiled_func = profile(f)
                result = profiled_func(self, *args, **kwargs)
            finally:
                # Restore original stdout
                sys.stdout = original_stdout
                # Store the captured profiling output in the class's memory_profiles dictionary
                profile_output = buf.getvalue()
                # Safe cast: memory_profiles is a dict
                cast(dict, memory_profiles)[f.__name__] = profile_output

                # Only try to extract memory usage if we have valid profiler output
                if profile_output.strip():
                    try:
                        current_memory: float = extract_final_memory_usage(
                            profile_output
                        )
                        logger.debug(
                            f"Current memory usage for {f.__name__}: {current_memory:.6f} MiB"
                        )

                        # Calculate memory delta if not the baseline method
                        if f.__name__ != baseline_method_name:
                            # Safe cast: memory_baseline is a float
                            baseline_memory = cast(
                                float, get_or_set_memory_attribute("memory_baseline")
                            )

                            # If baseline memory is not set yet, extract it from the baseline profile
                            if (
                                baseline_memory == 0.0
                                # Safe cast: memory_profiles is a dict
                                and baseline_method_name in cast(dict, memory_profiles)
                            ):
                                try:
                                    baseline_memory = extract_final_memory_usage(
                                        # Safe cast: memory_profiles is a dict
                                        cast(dict, memory_profiles)[
                                            baseline_method_name
                                        ]
                                    )
                                    # Validate against maximum allowed baseline memory limit
                                    if baseline_memory > MAX_BASELINE_MEMORY:
                                        raise AssertionError(
                                            f"Baseline memory footprint ({baseline_memory:.6f} MiB) "
                                            f"exceeds the maximum allowed limit of {MAX_BASELINE_MEMORY} MiB"
                                        )
                                    # Store the baseline memory for future use
                                    get_or_set_memory_attribute(
                                        "memory_baseline", baseline_memory
                                    )
                                except ValueError as e:
                                    raise ValueError(
                                        f"Could not extract baseline memory usage from {baseline_method_name} "
                                        f"profiler output. Original error: {e}"
                                    ) from e

                            # Calculate cumulative memory deltas from all previous tests
                            cumulative_previous_deltas = 0.0
                            # Safe cast: memory_deltas is a dict
                            for prev_method_name, prev_delta in cast(
                                dict, memory_deltas
                            ).items():
                                if prev_method_name != f.__name__:
                                    cumulative_previous_deltas += prev_delta

                            # Calculate delta: current - baseline - cumulative previous deltas
                            memory_delta = (
                                current_memory
                                - baseline_memory
                                - cumulative_previous_deltas
                            )

                            # Store this test's delta for future calculations
                            # Safe cast: memory_deltas is a dict
                            cast(dict, memory_deltas)[f.__name__] = memory_delta

                            logger.info(
                                f"~~~ Memory delta for {f.__name__}: {memory_delta:.6f} MiB "
                                f"(current: {current_memory:.6f}, baseline: {baseline_memory:.6f}, "
                                f"previous cumulative deltas: {cumulative_previous_deltas:.6f})"
                            )

                            # Check against memory limit
                            if memory_delta > max_memory:
                                raise AssertionError(
                                    f"Memory delta for {f.__name__} ({memory_delta:.6f} MiB) "
                                    f"exceeds the maximum allowed limit of {max_memory} MiB"
                                )
                        else:
                            # For the baseline method, extract and store the baseline memory
                            baseline_memory = current_memory
                            # Validate against maximum allowed baseline memory limit
                            if baseline_memory > MAX_BASELINE_MEMORY:
                                raise AssertionError(
                                    f"Baseline memory footprint ({baseline_memory:.6f} MiB) "
                                    f"exceeds the maximum allowed limit of {MAX_BASELINE_MEMORY} MiB"
                                )
                            get_or_set_memory_attribute(
                                "memory_baseline", baseline_memory
                            )
                            # For the baseline method, delta is 0
                            # Safe cast: memory_deltas is a dict
                            cast(dict, memory_deltas)[f.__name__] = 0.0

                    except ValueError as e:
                        # If profiler output is malformed, re-raise with more context
                        raise ValueError(
                            f"Could not extract memory usage from profiler output for {f.__name__}. "
                            f"Original error: {e}"
                        ) from e

            # Output the current memory usage statistics in each test method
            baseline_memory = cast(
                float, get_or_set_memory_attribute("memory_baseline")
            )
            memory_deltas = cast(
                dict[str, float], get_or_set_memory_attribute("memory_deltas")
            )
            output_current_memory_usage_stats(
                baseline_memory, memory_deltas, max_memory
            )

            return result

        return wrapper  # type: ignore[return-value]  # functools.wraps changes the wrapper type

    if func is None:
        # Called with parameters: e.g. `@memory_profile_and_capture(max_memory=1.0)`
        return decorator
    else:
        # Called without parameters: `@memory_profile_and_capture`
        return decorator(func)


def extract_final_memory_usage(profile_output: str) -> float:
    """
    Parses the memory_profiler output string and returns the final memory usage in MiB.
    :param profile_output: The output string from memory_profiler
    :type profile_output: str
    :return: The final memory usage in MiB
    :rtype: float
    :raises ValueError: If no memory usage data is found in the profile output.
    """
    # Check if memory profiling is disabled
    if not is_memory_profiling_enabled():
        logger.debug(
            "Memory profiling disabled - returning 0.0 for memory usage extraction"
        )
        return 0.0

    mem_usage_lines = []

    for line in profile_output.strip().splitlines():
        # Match lines that look like memory profiler data
        match = re.match(r"\s*(\d+)\s+([\d.]+)\s+MiB", line)
        if match:
            mem_usage = float(match.group(2))
            mem_usage_lines.append(mem_usage)

    if mem_usage_lines:
        return mem_usage_lines[-1]  # last recorded memory usage
    else:
        raise ValueError("No memory usage data found in profile output.")


def output_current_memory_usage_stats(
    baseline_memory: float, memory_deltas: dict[str, float], max_memory: float
) -> None:
    """
    Output comprehensive memory usage statistics for all test methods.

    This function categorizes memory deltas and provides detailed reporting including:
    - Positive deltas (methods that increased memory usage)
    - Negative deltas (methods that freed memory)
    - Zero deltas (methods with minimal memory change)
    - Exceeded deltas (methods that violated memory limits)
    - Overall statistics for full transparency

    The function maintains complete transparency by including all deltas in the overall
    statistics while providing clear categorization for developer-friendly analysis.

    :param baseline_memory: The baseline memory footprint in MiB established by the
        baseline test method
    :type baseline_memory: float
    :param memory_deltas: Dictionary mapping test method names to their memory deltas
        in MiB. Deltas represent the additional memory used by each method relative
        to the baseline and previous methods
    :type memory_deltas: dict[str, float]
    :param max_memory: Maximum allowed memory delta per test method in MiB. Used to
        identify methods that exceed the memory limit
    :type max_memory: float
    :return: None - this function only logs statistics, doesn't return values
    :rtype: None

    Note: Negative deltas indicate methods that freed memory (due to garbage collection,
    memory optimization, or other cleanup). These are reported separately and clearly
    labeled as "freed memory" to provide context.
    """
    # Categorize deltas for better reporting
    positive_deltas = []  # Methods that increased memory
    negative_deltas = []  # Methods that decreased memory (freed memory)
    zero_deltas = []  # Methods with no significant change
    exceeded_deltas = []  # Methods that exceeded the limit

    for method_name, delta in memory_deltas.items():
        if method_name == "test_establish_memory_baseline":
            continue  # Skip baseline method

        if delta > max_memory:
            exceeded_deltas.append((method_name, delta))
        elif (
            delta > 0.001
        ):  # Small positive threshold to avoid tiny floating point differences
            positive_deltas.append((method_name, delta))
        elif delta < -0.001:  # Small negative threshold
            negative_deltas.append((method_name, delta))
        else:
            zero_deltas.append((method_name, delta))

    logger.info(f"Baseline memory: {baseline_memory} MiB")

    # Report statistics for positive deltas (most relevant for limit checking)
    if positive_deltas:
        pos_values = [delta for _, delta in positive_deltas]
        min_pos = min(pos_values)
        avg_pos = sum(pos_values) / len(pos_values)
        max_pos = max(pos_values)

        logger.info(
            f"Positive deltas - Min: {min_pos:.6f} MiB, "
            f"Avg: {avg_pos:.6f} MiB, Max: {max_pos:.6f} MiB "
            f"({len(positive_deltas)} methods)"
        )

    # Report negative deltas (methods that freed memory)
    if negative_deltas:
        neg_values = [delta for _, delta in negative_deltas]
        logger.info(
            f"Memory freed - Total: {sum(neg_values):.6f} MiB "
            f"({len(negative_deltas)} methods)"
        )
        # Show which methods freed the most memory
        for method_name, delta in sorted(negative_deltas, key=lambda x: x[1]):
            logger.info(f"  • {method_name}: {delta:.6f} MiB (freed memory)")

    # Report methods with no significant change
    if zero_deltas:
        logger.info(f"Methods with minimal memory change: {len(zero_deltas)}")

    # Show methods that exceed the max delta per test value
    if exceeded_deltas:
        logger.warning(f"Methods exceeding max delta per test ({max_memory:.6f} MiB):")
        for method_name, delta in sorted(
            exceeded_deltas, key=lambda x: x[1], reverse=True
        ):
            logger.warning(f"  • {method_name}: {delta:.6f} MiB")
    else:
        logger.info("No methods exceeded the max delta per test limit")

    # Report overall statistics including all deltas for full transparency
    all_non_baseline_deltas = [
        delta
        for name, delta in memory_deltas.items()
        if name != "test_establish_memory_baseline"
    ]
    if all_non_baseline_deltas:
        logger.info(
            f"Overall - Min: {min(all_non_baseline_deltas):.6f} MiB, "
            f"Avg: {sum(all_non_baseline_deltas) / len(all_non_baseline_deltas):.6f} MiB, "
            f"Max: {max(all_non_baseline_deltas):.6f} MiB"
        )
