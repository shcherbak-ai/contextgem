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
Module defining utility functions for dynamically computing LLM output validation structures.
"""

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
    return type(name, (RootModel[root_type],), {})
