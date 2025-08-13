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

"""
Internal registry and utilities for connecting internal logic to the
public API surface.

This module implements a registry-based mapping system where:
- Internal classes (e.g. _Aspect) contain implementation
- Public classes (e.g. Aspect) provide stable API surface
- Registry ensures consistent object creation and deserialization

Mappings are declared exclusively via the ``_expose_in_registry`` class
decorator, which:

- Always registers the decorated class against itself (``cls -> cls``)
- Optionally accepts ``additional_key`` to also map another class to the
  decorated class (``additional_key -> cls``)

These mappings enable two core operations:

- Constructing public instances from implementation types via
  ``_publicize(source_type, **data)``
- Resolving public API types by class or name during (de)serialization via
  ``_resolve_public_type``
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel


_TModel = TypeVar("_TModel", bound=BaseModel)

# Maps classes to their public API counterparts.
_TYPE_REGISTRY: dict[type, type] = {}


def _resolve_public_type(source_type: type[_TModel] | str) -> type[_TModel]:
    """
    Resolve the registered public class for the given ``source_type``.

    :param source_type: Class to resolve.
    :type source_type: type[_TModel] | str
    :return: Mapped class for ``source_type`` if registered.
    :rtype: type[_TModel]
    :raises RuntimeError: If ``source_type`` does not have a registered mapping.
    """

    if isinstance(source_type, str):
        for internal_cls, public_cls in _TYPE_REGISTRY.items():
            if (
                internal_cls.__name__ == source_type
                or f"{internal_cls.__module__}.{internal_cls.__name__}" == source_type
            ):
                return public_cls
        raise RuntimeError(
            f"No public class mapping registered for type name: {source_type!r}"
        )

    mapped = _TYPE_REGISTRY.get(source_type)
    if mapped is None:
        raise RuntimeError(
            f"No public class mapping registered for type: {source_type!r}"
        )
    return mapped


def _publicize(source_type: type[_TModel], /, **data: Any) -> _TModel:
    """
    Construct and return an instance of the registered public class
    that is mapped to the given ``source_type``.

    :param source_type: Class to publicize (e.g., ``_StringConcept``).
    :type source_type: type[_TModel]
    :param data: Keyword arguments to construct the instance.
    :type data: Any
    :return: A new instance of the mapped class.
    :rtype: _TModel
    """

    target_cls = _resolve_public_type(source_type)
    return target_cls(**data)


def _expose_in_registry(
    cls: type[_TModel] | None = None,
    *,
    additional_key: type[_TModel] | None = None,
) -> type[_TModel] | Any:
    """
    Class decorator to mark classes as exposed on the public API surface by registering
    a self-mapping in the type registry.

    Always registers the decorated class as ``cls -> cls``.

    Optionally, ``additional_key`` can be provided to additionally register that class
    to the decorated class. This is useful when more than one class needs to be mapped
    against the decorated class for deserialization purposes.

    :param cls: The class being decorated (automatically supplied when used as a decorator).
    :type cls: type[_TModel] | None
    :param additional_key: An additional class to map to the decorated class in the registry.
    :type additional_key: type[_TModel] | None
    :return: The original class when used as a decorator, or a decorator function when used with arguments.
    :rtype: type[_TModel] | Callable[[type[_TModel]], type[_TModel]]
    """

    def _apply(target_cls: type[_TModel]) -> type[_TModel]:
        """
        Register the decorated class and optional alias in the type registry.

        Always registers ``target_cls -> target_cls``. If the outer decorator was
        called with ``additional_key``, also registers ``additional_key -> target_cls``.

        :param target_cls: The class being decorated and registered.
        :type target_cls: type[_TModel]
        :return: The same class to enable decorator chaining.
        :rtype: type[_TModel]
        """
        # Register the decorated class against itself
        _TYPE_REGISTRY[target_cls] = target_cls

        # Optionally register the extra class against the decorated class
        if additional_key is not None:
            _TYPE_REGISTRY[additional_key] = target_cls

        return target_cls

    # Support both bare decorator and decorator with arguments
    if cls is None:
        return _apply
    return _apply(cls)
