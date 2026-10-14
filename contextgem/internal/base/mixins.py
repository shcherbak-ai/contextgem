# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining base mixin classes for extending functionality through class composition.

This module provides mixin classes that can be used to add specific behaviors to other classes
through multiple inheritance. Currently includes the _PostInitCollectorMixin which enables
post-initialization processing for Pydantic models, the _AssignmentRollbackMixin which
keeps a Pydantic model unchanged when an assignment to its field is rejected, and
the _ApiKeyRedactionMixin which keeps the API key out of the validation errors of
a Pydantic model that has one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, get_args

from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError
from pydantic_core.core_schema import ErrorType


# The name of the param that holds the API key
_API_KEY_PARAM = "api_key"

# What a value that is, or may be, an API key is replaced with in the inputs
# that a validation error reports
_REDACTED_VALUE = "**********"

# The types of the errors that reject a value passed in a param that does not
# exist: when an instance is created, and when the value is assigned to it.
# The param is the last item of the location of such an error.
_UNKNOWN_PARAM_ERROR_TYPES = frozenset({"extra_forbidden", "no_such_attribute"})

# The types of the errors that Pydantic defines, which are rebuilt from the type
# and the context. An error of any other type is a custom one, which is rebuilt
# from its message, as is a custom error that is given one of these types.
_PYDANTIC_ERROR_TYPES: frozenset[str] = frozenset(get_args(ErrorType))


def _redact_values_of_params(
    value: Any, params: set[int | str], parent_ids: frozenset[int] = frozenset()
) -> Any:
    """
    Returns the value with the values of the given params replaced with
    the redaction placeholder, in the dicts that the value is or holds.

    The dicts (and other mappings), lists and tuples are looked through at any
    depth. Nothing is changed in place: a dict, a list or a tuple that holds
    a redacted value is replaced with its copy, and any other value is returned as
    it is, i.e. as the same object. A value of another type is not looked into,
    e.g. a model is returned as it is, with its own API key as its attribute.
    A param that is set to None has no value to redact.

    :param value: The value to redact the values of the params in.
    :type value: Any
    :param params: The names of the params whose values are redacted.
    :type params: set[int | str]
    :param parent_ids: The identifiers of the dicts, lists and tuples that hold
        the value, i.e. that are being looked through. Defaults to none.
    :type parent_ids: frozenset[int]
    :return: The value without the values of the params.
    :rtype: Any
    """
    if not isinstance(value, Mapping | list | tuple):
        return value
    if id(value) in parent_ids:
        # The value holds itself. Looking through it again would never end, and
        # as it is, it has the values that are redacted in its copy.
        return _REDACTED_VALUE
    parent_ids = parent_ids | {id(value)}
    if isinstance(value, Mapping):
        redacted_dict = {
            name: (
                _REDACTED_VALUE
                if name in params and item is not None
                else _redact_values_of_params(item, params, parent_ids)
            )
            for name, item in value.items()
        }
        if all(redacted_dict[name] is item for name, item in value.items()):
            return value
        return redacted_dict
    redacted_items = [
        _redact_values_of_params(item, params, parent_ids) for item in value
    ]
    if all(
        redacted_item is item
        for redacted_item, item in zip(redacted_items, value, strict=True)
    ):
        return value
    return redacted_items if isinstance(value, list) else tuple(redacted_items)


def _redact_api_key_in_validation_error(
    error: ValidationError, hide_input: bool
) -> ValidationError:
    """
    Returns a copy of the validation error of a model that does not give
    the API key in the inputs that it reports, i.e. in what its ``errors()`` and
    ``json()`` methods return.

    The API key is the value of the ``api_key`` param. The value of a param that
    the model does not have is treated in the same way, whatever it is, as it may
    be the key passed under a misspelled name, or to a model that has no such
    param. Such a value is replaced with the redaction placeholder wherever
    the error reports it:

    - as the input of an error for that param, e.g. a key of a wrong type;
    - as an item of an input that is, or holds, the params that are passed as
      a dict, e.g. all the params that a model is created with, which are the input
      of a check that involves several of them and of an error for a required
      param that is not passed.

    Everything else is the same as in the original error: the title, the number
    and the order of the errors, their types, locations, messages, contexts and
    URLs, and the other inputs. A model that is among the inputs is returned as it
    is, i.e. with its own API key as its attribute: a model is printed and
    serialized without the key (see the ``api_key`` field of ``_DocumentModel``).
    An API key that is passed as the value of another param of the model cannot
    be told from any other value, and is not redacted.

    :param error: The validation error.
    :type error: ValidationError
    :param hide_input: Whether the text of the error does not show the inputs,
        as set for the model with ``hide_input_in_errors``.
    :type hide_input: bool
    :return: The validation error without the API key in its inputs.
    :rtype: ValidationError
    """
    errors = error.errors()
    # The params whose values are redacted: `api_key`, and the params that
    # the model does not have
    redacted_params: set[int | str] = {_API_KEY_PARAM}
    for details in errors:
        if details["type"] in _UNKNOWN_PARAM_ERROR_TYPES:
            redacted_params.update(details["loc"][-1:])
    line_errors: list[InitErrorDetails] = []
    for details in errors:
        if redacted_params.intersection(details["loc"]):
            # The input is the value of such a param, or a part of the value
            input_value = None if details["input"] is None else _REDACTED_VALUE
        else:
            input_value = _redact_values_of_params(details["input"], redacted_params)
        error_type: str | PydanticCustomError = details["type"]
        context = details.get("ctx")
        if error_type not in _PYDANTIC_ERROR_TYPES or "url" not in details:
            # A custom error, which is also what an error of a type of Pydantic
            # is if it is reported without a URL. It has no template of its
            # message to be rebuilt with: its message, with the values of
            # the context filled in, is the template.
            error_type = PydanticCustomError(
                details["type"],  # ty: ignore[invalid-argument-type]
                details["msg"],  # ty: ignore[invalid-argument-type]
                context,
            )
        line_error: InitErrorDetails = {
            "type": error_type,
            "loc": details["loc"],
            "input": input_value,
        }
        if context is not None:
            line_error["ctx"] = context
        line_errors.append(line_error)
    return ValidationError.from_exception_data(
        error.title, line_errors, hide_input=hide_input
    )


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


class _ApiKeyRedactionMixin(BaseModel):
    """
    A mixin class that keeps the API key out of the validation errors of an instance.

    The inputs of a failed validation are reported by the validation error: they
    are returned by its ``errors()`` method and written out by its ``json()``
    method. The API key is among them when it is rejected itself, e.g. a key of
    a wrong type or a key passed under a misspelled name, and whenever the input
    is all the params that the instance is created with. With this mixin,
    a validation error of the instance is raised with such values replaced with
    a placeholder (see ``_redact_api_key_in_validation_error()``), and is the same
    otherwise.

    This applies wherever the instance is validated: when it is created, which
    is also how it is deserialized, and when a value is assigned to its field.
    """

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as error:
            redacted_error = self._redact_api_key_in_validation_error(error)
        else:
            return
        # Raised here, and not while the original error is handled: the original
        # error, which reports the key, would be attached to this one as its context
        raise redacted_error

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Assigns the value to the attribute. If the value is assigned to a field and
        is rejected, the validation error is raised without the API key in its inputs.

        :param name: The name of the attribute to set.
        :type name: str
        :param value: The value to assign to the attribute.
        :type value: Any
        :return: None
        :rtype: None
        """
        try:
            super().__setattr__(name, value)
        except ValidationError as error:
            redacted_error = self._redact_api_key_in_validation_error(error)
        else:
            return
        # Raised here for the same reason as when the instance is created
        raise redacted_error

    def _redact_api_key_in_validation_error(
        self, error: ValidationError
    ) -> ValidationError:
        """
        Returns a copy of the validation error of the instance that does not give
        the API key in the inputs that it reports. The text of the copy shows
        the inputs, or does not show them, as set for the class of the instance.

        :param error: The validation error of the instance.
        :type error: ValidationError
        :return: The validation error without the API key in its inputs.
        :rtype: ValidationError
        """
        return _redact_api_key_in_validation_error(
            error,
            hide_input=bool(type(self).model_config.get("hide_input_in_errors")),
        )
