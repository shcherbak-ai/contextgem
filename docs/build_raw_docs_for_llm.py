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
Script for building a single text file containing all ContextGem documentation.

This script is essential for preparing documentation for LLM ingestion, particularly
for enabling user Q&A functionality. It performs the following steps:
1. Builds Sphinx documentation in text format
2. Extracts document paths from the index.rst toctree
3. Concatenates all documentation files into a single text file

The resulting file (docs/docs-raw-for-llm.txt) serves as the knowledge base for
LLM-powered documentation queries and assistance.

To use it, run:

```bash
python docs/build_raw_docs_for_llm.py
```
"""

import logging
import os
import subprocess
import sys


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SOURCE_DIR = "docs/source"
TEXT_DIR = "docs/build/text"
INDEX_PATH = os.path.join(SOURCE_DIR, "index.rst")
OUTPUT_PATH = "docs/docs-raw-for-llm.txt"


def run_sphinx_text_build() -> None:
    """Run Sphinx build in text format to generate plain text documentation.

    Raises:
        SystemExit: If the Sphinx build process fails.
    """
    logger.info("🔧 Running: sphinx-build -b text")
    try:
        # Run sphinx-build with full path and environment
        subprocess.run(
            ["sphinx-build", "-b", "text", SOURCE_DIR, TEXT_DIR, "-E"],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        logger.info("✅ Sphinx text build completed.")
    except subprocess.CalledProcessError as e:
        logger.error("❌ Failed to build Sphinx docs (text format):")
        logger.error(f"Exit code: {e.returncode}")
        logger.error("stdout:")
        logger.error(e.stdout)
        logger.error("stderr:")
        logger.error(e.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during Sphinx build: {e}")
        sys.exit(1)


def extract_doc_paths(index_path: str) -> list[str]:
    """Extract document paths from the index.rst file's toctree directives.

    Args:
        index_path (str): Path to the index.rst file.

    Returns:
        list[str]: List of document paths found in the toctree directives.
    """
    doc_paths = []
    in_toctree = False
    current_block = []

    with open(index_path, encoding="utf-8") as f:
        lines = f.readlines()

    for _i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(".. toctree::"):
            if current_block:
                doc_paths.extend(current_block)
                current_block = []
            in_toctree = True
            continue

        if in_toctree:
            if stripped.startswith(":"):
                continue  # toctree options
            elif stripped.startswith(".. "):
                # Next directive starts, flush previous block
                if current_block:
                    doc_paths.extend(current_block)
                    current_block = []
                in_toctree = False
            elif stripped == "":
                continue  # skip blank lines
            else:
                current_block.append(stripped)

    # End of file — flush remaining entries
    if current_block:
        doc_paths.extend(current_block)

    return doc_paths


def concatenate_docs(doc_paths: list[str], text_dir: str, output_file: str) -> None:
    """Concatenate multiple documentation files into a single output file.

    Args:
        doc_paths (list[str]): List of document paths to concatenate.
        text_dir (str): Directory containing the text documentation files.
        output_file (str): Path to the output file where concatenated docs will be saved.
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("ContextGem - Effortless LLM extraction from documents\n")
        outfile.write("=" * 100 + "\n\n")
        outfile.write("Copyright (c) 2025 Shcherbak AI AS\n")
        outfile.write("All rights reserved\n")
        outfile.write("Developed by Sergii Shcherbak\n\n")
        outfile.write(
            'This software is licensed under the Apache License, Version 2.0 (the "License");\n'
        )
        outfile.write(
            "you may not use this file except in compliance with the License.\n"
        )
        outfile.write("You may obtain a copy of the License at\n\n")
        outfile.write("     http://www.apache.org/licenses/LICENSE-2.0\n\n")
        outfile.write(
            "Unless required by applicable law or agreed to in writing, software\n"
        )
        outfile.write(
            'distributed under the License is distributed on an "AS IS" BASIS,\n'
        )
        outfile.write(
            "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        )
        outfile.write(
            "See the License for the specific language governing permissions and\n"
        )
        outfile.write("limitations under the License.\n\n")
        outfile.write("# ==== Documentation Content ====\n\n")
        for doc in doc_paths:
            txt_file = os.path.join(text_dir, f"{doc}.txt")
            if os.path.exists(txt_file):
                outfile.write(f"\n\n# ==== {doc} ====\n\n")
                with open(txt_file, encoding="utf-8") as infile:
                    outfile.write(infile.read())
            else:
                logger.info(f"⚠️  Warning: {txt_file} not found. Skipping.")


if __name__ == "__main__":
    try:
        run_sphinx_text_build()
        paths = extract_doc_paths(INDEX_PATH)
        if not paths:
            logger.error("❌ No document paths found in index.rst toctree blocks.")
            sys.exit(1)

        logger.info("📄 Document order extracted:")
        for p in paths:
            logger.info(f"   - {p}")

        concatenate_docs(paths, TEXT_DIR, OUTPUT_PATH)
        logger.info(f"\n✅ Concatenated documentation saved to {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
