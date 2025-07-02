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
Module defining base mixin classes for extending functionality through class composition.

This module provides mixin classes that can be used to add specific behaviors to other classes
through multiple inheritance. Currently includes the _PostInitCollectorMixin which enables
post-initialization processing for Pydantic models.
"""

from typing import Any, Callable

from pydantic import BaseModel


class _PostInitCollectorMixin(BaseModel):
    """
    A mixin class that collects, holds, and executes post-initialization methods.

    This mixin uses Method Resolution Order (MRO) to gather all methods marked with
    the `__post_init__` attribute (e.g. via a decorator) from the inheritance hierarchy.
    These methods are then executed in order during the initialization process, allowing for
    customized post-initialization behavior across multiple inheritance levels.

    :ivar __post_init_methods__: List of callable methods to be executed after initialization.
    :vartype __post_init_methods__: list[Callable[[Any, Any], None]]
    """

    # Holds all post-init methods for the class.
    __post_init_methods__: list[Callable[[Any, Any], None]] = []

    def model_post_init(self, __context: Any) -> None:
        """
        Executes the post-init methods defined in each base class.

        :param __context: The context to be passed to each post-init method.
        """
        # Pydantic-specific
        for func in self.__class__.__post_init_methods__:
            func(self, __context)

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass-specific behavior and track methods marked with the
        __post_init__ attribute (e.g. via a decorator) in a class-level registry.

        :param cls: The newly created subclass.
        :param kwargs: Additional keyword arguments passed during the subclass
            initialization process. These are forwarded to the superclass's
            ``__init_subclass__`` implementation.
        """
        super().__init_subclass__(**kwargs)
        methods: list[Callable[[Any, Any], None]] = []
        seen_ids = set()
        for base in cls.__mro__:
            for attr, value in base.__dict__.items():
                if callable(value) and getattr(value, "__post_init__", False):
                    # Use the function's identity to ensure that two distinct functions
                    # with the same name (from different classes) are both added.
                    if id(value) not in seen_ids:
                        methods.append(value)
                        seen_ids.add(id(value))
        cls.__post_init_methods__ = methods
