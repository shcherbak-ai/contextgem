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
Module defining public utility functions of the framework.
"""

import base64
from pathlib import Path

from contextgem.internal.loggers import _configure_logger_from_env


def image_to_base64(image_path: str | Path) -> str:
    """
    Converts an image file to its Base64 encoded string representation.

    Helper function that can be used when constructing ``Image`` objects.

    :param image_path: The path to the image file to be encoded.
    :type image_path: str | Path
    :return: A Base64 encoded string representation of the image.
    :rtype: str
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def reload_logger_settings():
    """
    Reloads logger settings from environment variables.

    This function should be called when environment variables related to logging
    have been changed after the module was imported. It re-reads the environment
    variables and reconfigures the logger accordingly.

    :return: None

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/utils/reload_logger_settings.py
            :language: python
            :caption: Reload logger settings
    """
    _configure_logger_from_env()
