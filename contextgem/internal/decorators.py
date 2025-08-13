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
Module defining internal decorators for the framework.
"""

from __future__ import annotations

import asyncio
import timeit
from collections.abc import Callable
from functools import wraps
from typing import Any

from contextgem.internal.loggers import logger
from contextgem.internal.registry import _expose_in_registry as _expose_in_registry


def _post_init_method(func: Callable[[Any, Any], None]) -> Callable[[Any, Any], None]:
    """
    Decorates a given function to mark it as a post-initialization method.
    To be used with method names other than pydantic-specific `model_post_init`, to collect them
    and execute at once for all subclasses with a long MRO.

    :param func: The target function to be decorated and flagged as a
        post-initialization method.
    :type func: Callable[[Any, Any], None]

    :return: The same function provided via the `func` parameter, now marked
        as a post-initialization method by setting `__post_init__`.
    :rtype: Callable[[Any, Any], None]
    """
    # Dynamically mark function as a post-init method for MRO collection
    func.__post_init__ = True  # type: ignore[attr-defined]
    return func


def _timer_decorator(
    process_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to measure execution time for both sync and async functions.

    :param process_name: The name of the process to be logged and measured.
    :return: A decorator function that wraps the provided function to log its execution time.
    """

    def outer_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Outer wrapper function that creates appropriate timer wrapper based on function type.

        :param func: The function to be wrapped with timing functionality.
        :return: Either async_wrapper or sync_wrapper depending on the function type.
        """

        def log_end(start_time: float) -> None:
            """
            Logs the end time and duration of a process.

            :param start_time: The start time of the process.
            """
            end_time = timeit.default_timer()
            elapsed_time = round(end_time - start_time, 2)
            logger.success(
                f"Process `{process_name}` finished in {elapsed_time} seconds."
            )

        if asyncio.iscoroutinefunction(func):
            # Async wrapper
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """
                Async wrapper that measures execution time for async functions.

                :param args: Positional arguments to pass to the wrapped function.
                :param kwargs: Keyword arguments to pass to the wrapped function.
                :return: The result of the wrapped function execution.
                """
                start_time = timeit.default_timer()
                result = await func(*args, **kwargs)
                log_end(start_time)
                return result

            return async_wrapper

        else:
            # Sync wrapper
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """
                Sync wrapper that measures execution time for synchronous functions.

                :param args: Positional arguments to pass to the wrapped function.
                :param kwargs: Keyword arguments to pass to the wrapped function.
                :return: The result of the wrapped function execution.
                """
                start_time = timeit.default_timer()
                result = func(*args, **kwargs)
                log_end(start_time)
                return result

            return sync_wrapper

    return outer_wrapper


def _disable_direct_initialization(cls: type) -> type:
    """
    Class decorator that disables direct instantiation of the decorated class.

    Used to ensure that certain internal classes are not instantiated directly and public API
    returns only the intended classes.

    Usage:
        @_prevent_direct_initialization
        class _InternalCls:
            ...

        # Raises TypeError
        _InternalCls()

        # Subclasses can still be instantiated (unless they override ``__new__``
        # to re-use the guard from the parent):
        class PublicSub(_InternalCls):
            pass

        PublicSub()  # OK

    The guard is implemented by overriding ``__new__`` on the exact decorated
    class only. Subclasses are allowed to instantiate normally.

    :param cls: Class to protect from direct instantiation.
    :return: The same class with guarded ``__new__``.
    """

    def _guarded_new(inner_cls: type, *args: Any, **kwargs: Any) -> Any:
        """
        Replacement for ``__new__`` that blocks direct instantiation of the
        decorated class while allowing subclasses to instantiate normally.

        If invoked for the exact decorated class, an error is logged and
        ``TypeError`` is raised. For subclasses, the call is forwarded to the
        next ``__new__`` in the MRO, preserving base-class behavior (e.g.,
        Pydantic's ``BaseModel.__new__``).

        :param inner_cls: The class being instantiated (decorated class or
            its subclass).
        :param args: Positional constructor arguments.
        :param kwargs: Keyword constructor arguments.
        :return: A new instance when called for a subclass.
        :raises TypeError: When attempting to instantiate the decorated class
            directly.
        """
        if inner_cls is cls:
            class_name = getattr(inner_cls, "__name__", str(inner_cls))
            logger.error(
                f"Attempted to instantiate internal class `{class_name}` directly."
            )
            raise TypeError(
                f"Class `{class_name}` is internal and cannot be instantiated directly. "
                f"Use the appropriate public factory or API instead."
            )
        # Allow normal instantiation for subclasses using the next __new__ in MRO
        # relative to the decorated class. This preserves behaviors of bases such
        # as Pydantic's BaseModel.__new__.
        base_new = super(cls, inner_cls).__new__
        try:
            return base_new(inner_cls, *args, **kwargs)
        except TypeError:
            # Fallback for bases like object.__new__ that don't accept args
            return base_new(inner_cls)

    # Bind as __new__ on the class so the guard triggers only for the exact class
    cls.__new__ = staticmethod(_guarded_new)
    return cls
