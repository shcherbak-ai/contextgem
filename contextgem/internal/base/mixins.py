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
post-initialization processing for Pydantic models, and the _AssignmentRollbackMixin which
keeps a Pydantic model unchanged when an assignment to its field is rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class _PostInitCollectorMixin(BaseModel):
    """
    A mixin class that collects, holds, and executes post-initialization methods.

    This mixin uses Method Resolution Order (MRO) to gather all methods marked with
    the `__post_init__` attribute (e.g. via a decorator) from the inheritance hierarchy.
    These methods are then executed in order during the initialization process, allowing for
    customized post-initialization behavior across multiple inheritance levels.

    :ivar __post_init_methods__: List of callable methods to be executed after initialization.
    :vartype __post_init_methods__: list[Callable[[Any, Any], Any]]
    """

    # Holds all post-init methods for the class.
    __post_init_methods__: list[Callable[[Any, Any], Any]] = []

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
        methods: list[Callable[[Any, Any], Any]] = []
        seen_ids = set()
        for base in cls.__mro__:
            for _attr, value in base.__dict__.items():
                if (
                    callable(value)
                    and getattr(value, "__post_init__", False)
                    and id(value) not in seen_ids
                ):
                    # Use the function's identity to ensure that two distinct functions
                    # with the same name (from different classes) are both added.
                    methods.append(value)
                    seen_ids.add(id(value))
        cls.__post_init_methods__ = methods


class _AssignmentRollbackMixin(BaseModel):
    """
    A mixin class that keeps an instance unchanged when an assignment to its field
    is rejected.

    With ``validate_assignment=True``, Pydantic calls the validators of the model that
    run after the validation of the fields (``@model_validator(mode="after")``) when
    the new value is already assigned. If such a validator rejects the value, the error
    is raised, but the rejected value is kept, i.e. the instance is left in the state
    that the error reports as invalid. This mixin restores the fields of the instance
    as they were before the assignment.

    Only the fields of the instance are restored. A validator that derives other state
    from the fields, e.g. private attributes, is to update that state after all its
    checks pass, so that there is nothing else to restore.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Assigns the value to the attribute, restoring the fields of the instance
        if the value is assigned to a field and is rejected.

        :param name: The name of the attribute to set.
        :type name: str
        :param value: The value to assign to the attribute.
        :type value: Any
        :return: None
        :rtype: None
        """
        if name not in type(self).model_fields:
            # Private attributes and properties are not validated by the model
            super().__setattr__(name, value)
            return

        fields_snapshot = self._get_fields_snapshot()
        try:
            super().__setattr__(name, value)
        except Exception:
            self._restore_fields(fields_snapshot)
            raise

    def _get_fields_snapshot(self) -> tuple[dict[str, Any], set[str]]:
        """
        Returns the current values of the fields of the instance, to be restored with
        ``_restore_fields()`` if an assignment that follows is rejected.

        The values are not copied, i.e. the snapshot does not cover the changes made
        to a value in place.

        :return: The values of the fields, and the names of the fields that were set
            explicitly.
        :rtype: tuple[dict[str, Any], set[str]]
        """
        return self.__dict__.copy(), self.__pydantic_fields_set__.copy()

    def _restore_fields(self, fields_snapshot: tuple[dict[str, Any], set[str]]) -> None:
        """
        Restores the fields of the instance from the snapshot returned by
        ``_get_fields_snapshot()``, without validation: the snapshot is taken from
        the instance, i.e. its values are already validated.

        :param fields_snapshot: The snapshot of the fields to restore.
        :type fields_snapshot: tuple[dict[str, Any], set[str]]
        :return: None
        :rtype: None
        """
        fields, fields_set = fields_snapshot
        self.__dict__.clear()
        self.__dict__.update(fields)
        self.__pydantic_fields_set__.clear()
        self.__pydantic_fields_set__.update(fields_set)
