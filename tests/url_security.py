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
URL security validation for VCR test cassettes.

This module provides security validation for URLs used in VCR test recordings,
ensuring that only approved domains are accessed during tests and that existing
cassette files contain only authorized URLs.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

from contextgem.internal.loggers import logger
from tests.conftest import VCR_DUMMY_ENDPOINT_PREFIX


# API URL domains used during tests, before any URL redaction occurs
APPROVED_URL_DOMAINS_BEFORE_RECORDING = {
    "openai.azure.com",  # Azure OpenAI
    "api.openai.com",  # OpenAI
    "localhost:11434",  # Ollama
    "localhost:1234",  # LM Studio
    "huggingface.co",  # Hugging Face for loading SaT models
    "hf.co",  # Hugging Face for loading SaT models
}

# Cache for domain validation results to avoid repeated parsing and validation
_DOMAIN_VALIDATION_CACHE = {}


class URLSecurityError(Exception):
    """
    Exception raised when a URL security violation is detected,
    i.e. when a URL domain is not in the approved whitelist.
    """

    pass


def extract_domain_from_url(url: str) -> str:
    """
    Extract domain from URL for security validation.

    :param url: The URL to extract domain from
    :return: The extracted domain, empty string if parsing fails
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def is_url_approved(url: str) -> bool:
    """
    Check if URL domain is in approved list with caching.

    :param url: The URL to validate
    :return: True if URL domain is approved, False otherwise
    """
    # Check cache first
    if url in _DOMAIN_VALIDATION_CACHE:
        return _DOMAIN_VALIDATION_CACHE[url]

    domain = extract_domain_from_url(url)
    if not domain:
        result = False
    else:
        # Check exact match
        if domain in APPROVED_URL_DOMAINS_BEFORE_RECORDING:
            result = True
        else:
            # Check if domain is subdomain of approved domain
            result = any(
                domain.endswith("." + approved_domain)
                for approved_domain in APPROVED_URL_DOMAINS_BEFORE_RECORDING
            )

    # Cache the result
    _DOMAIN_VALIDATION_CACHE[url] = result
    return result


def validate_existing_cassettes_urls_security(
    cassettes_dir: str = "tests/cassettes",
) -> dict:
    """
    Checks that all request URLs in the cassette files are from approved domains.

    :param cassettes_dir: Directory containing cassette files
    :return: Dictionary with validation results containing
        'total_files', 'valid_files', and 'violations'
    :raises URLSecurityError: If any violations are found in the cassette files
    """

    results = {"total_files": 0, "valid_files": 0, "violations": []}

    cassettes_path = Path(cassettes_dir)
    if not cassettes_path.exists():
        logger.warning(f"Cassettes directory {cassettes_dir} does not exist")
        return results

    for cassette_file in cassettes_path.glob("*.yaml"):
        results["total_files"] += 1

        try:
            with open(cassette_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or "interactions" not in data:
                results["valid_files"] += 1
                continue

            file_violations = []
            for interaction in data["interactions"]:
                if "request" in interaction and "uri" in interaction["request"]:
                    uri = interaction["request"]["uri"]
                    if not uri.startswith(
                        VCR_DUMMY_ENDPOINT_PREFIX
                    ) and not is_url_approved(uri):
                        domain = extract_domain_from_url(uri)
                        file_violations.append({"url": uri, "domain": domain})

            if file_violations:
                results["violations"].append(
                    {"file": str(cassette_file), "violations": file_violations}
                )
            else:
                results["valid_files"] += 1

        except Exception as e:
            logger.error(f"Error reading cassette file {cassette_file}: {e}")
            results["violations"].append({"file": str(cassette_file), "error": str(e)})

    # Log violations if any were found
    if results["violations"]:
        logger.warning(
            f"Found {len(results['violations'])} files with URL security violations"
        )
        for violation in results["violations"]:
            if "error" in violation:
                logger.error(f"Error in file {violation['file']}: {violation['error']}")
            else:
                logger.warning(f"Security violations in {violation['file']}:")
                for v in violation["violations"]:
                    logger.warning(
                        f"  - Unapproved URL: {v['url']} (domain: {v['domain']})"
                    )

        # Raise security error after logging violations
        error_msg = f"URL security violations found in {len(results['violations'])} cassette files:\n"
        for violation in results["violations"]:
            if "error" in violation:
                error_msg += (
                    f"Error in file {violation['file']}: {violation['error']}\n"
                )
            else:
                error_msg += f"Security violations in {violation['file']}:\n"
                for v in violation["violations"]:
                    error_msg += (
                        f"  - Unapproved URL: {v['url']} (domain: {v['domain']})\n"
                    )

        raise URLSecurityError(error_msg)
    else:
        logger.info(
            f"All {results['valid_files']} cassette files passed URL security validation"
        )

    return results
