# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining utility functions for dynamically computing LLM output validation structures.
"""

from __future__ import annotations

from pydantic import RootModel


def _create_root_model(name: str, root_type: type):
    """
    Creates a dynamic model class extending RootModel for a specified type.

    :param name: The name of the new class to be created.
    :type name: str
    :param root_type: The root type to be used as a parameter for RootModel.
    :type root_type: type
    :return: A dynamically created class inheriting from RootModel
        parameterized with the given type.
    :rtype: type
    """
    return type(name, (RootModel[root_type],), {})  # ty: ignore[invalid-type-form]
