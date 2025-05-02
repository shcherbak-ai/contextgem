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
Exceptions for the DOCX converter module.

This module defines custom exception classes used by the DOCX converter
to handle various error conditions that may occur during document processing.
These exceptions provide more specific error information than generic exceptions,
making it easier to diagnose and handle problems when working with DOCX files.
"""


# Define custom exceptions
class DocxConverterError(Exception):
    """Base exception class for DOCX converter errors."""

    pass


class DocxFormatError(DocxConverterError):
    """Exception raised when the DOCX file format is invalid or corrupted."""

    pass


class DocxXmlError(DocxConverterError):
    """Exception raised when there's an error parsing XML in the DOCX file."""

    pass


class DocxContentError(DocxConverterError):
    """Exception raised when required content is missing from the DOCX file."""

    pass
