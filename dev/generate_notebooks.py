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
Script to generate Python notebooks from Python example files.
It simply pastes the entire content of each Python file into a notebook.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# Root directories
BASE_DIR = Path(__file__).parent
DOCS_INPUT_DIR = BASE_DIR / "usage_examples" / "docs"
README_INPUT_DIR = BASE_DIR / "usage_examples" / "readme"
DOCS_OUTPUT_DIR = BASE_DIR / "notebooks" / "docs"
README_OUTPUT_DIR = BASE_DIR / "notebooks" / "readme"

# Directories to exclude from notebook generation (full paths)
EXCLUDE_DIRS = [
    str(DOCS_INPUT_DIR / "optimizations"),
    str(DOCS_INPUT_DIR / "serialization"),
    str(DOCS_INPUT_DIR / "llms" / "llm_init"),
    str(DOCS_INPUT_DIR / "llm_config"),
    str(DOCS_INPUT_DIR / "concepts" / "json_object_concept" / "structure"),
]


def extract_first_comment(file_path: str | Path) -> str | None:
    """
    Extract comment lines from the beginning of a Python file to use as a title.
    Captures multiple consecutive comment lines.

    Args:
        file_path: Path to the Python file

    Returns:
        The combined comment lines if found, None otherwise
    """
    comment_lines = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            # Skip empty lines at the beginning
            if not line.strip():
                continue

            # Check if the line starts with a comment
            match = re.match(r"^#\s*(.*)", line.strip())
            if match:
                comment_lines.append(match.group(1))
            else:
                # Stop at the first non-comment line
                break

    if not comment_lines:
        return None

    # Join all comment lines with a space
    return " ".join(comment_lines)


def create_deterministic_metadata() -> dict[str, Any]:
    """
    Create deterministic metadata for notebooks.

    Returns:
        Dictionary with stable metadata
    """
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0",
        },
    }


def create_deterministic_cell_metadata() -> dict[str, Any]:
    """
    Create deterministic cell metadata.

    Returns:
        Dictionary with stable cell metadata
    """
    # Using empty dict - no cell metadata needed
    return {}


def create_notebook_from_file(file_path: str | Path, output_path: str | Path) -> None:
    """
    Create a Jupyter notebook from a Python file by pasting its entire content.
    Uses deterministic IDs and metadata to ensure consistent output.

    Args:
        file_path: Path to the Python file to convert
        output_path: Path where the notebook will be saved
    """
    # Read the entire file content
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    cells = []

    # Extract first comment for title, or use filename
    first_comment = extract_first_comment(file_path)
    if first_comment:
        title = first_comment
    else:
        title = (
            os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
        )

    # Add a title cell
    cells.append(new_markdown_cell(f"# {title}"))

    # Add pip install cell
    cells.append(new_code_cell("%pip install -U contextgem"))

    # Add instruction cell
    cells.append(
        new_markdown_cell(
            "To run the extraction, please provide your LLM details in the ``DocumentLLM(...)`` constructor further below."
        )
    )

    # Add the entire file content as a single code cell
    cells.append(new_code_cell(content))

    # Create the notebook
    nb = new_notebook(cells=cells, metadata=create_deterministic_metadata())

    # Remove any variable metadata at notebook level
    for key in [
        "creation_date",
        "modified_date",
        "timestamp",
        "date",
        "creation",
        "modified",
    ]:
        if key in nb.metadata:
            del nb.metadata[key]

    # Add deterministic cell IDs and clean cell metadata
    for i, cell in enumerate(nb.cells):
        # Set deterministic cell ID
        cell.id = f"cell_{i}"

        # Ensure execution_count is None for code cells
        if hasattr(cell, "execution_count") and cell.execution_count is not None:
            cell.execution_count = None

        # Remove any outputs for deterministic results
        if hasattr(cell, "outputs"):
            cell.outputs = []

        # Set deterministic cell metadata
        cell.metadata = create_deterministic_cell_metadata()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the notebook with deterministic ordering of keys
    with open(output_path, "w", encoding="utf-8") as f:
        # Use nbformat.writes + json.dump with sort_keys for deterministic output
        json_content = nbformat.writes(nb)
        parsed = json.loads(json_content)
        json.dump(parsed, f, sort_keys=True, indent=2)

    print(f"Created notebook: {output_path}")


def should_skip_file(file_path: str | Path) -> bool:
    """
    Check if the file should be skipped based on path.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file should be skipped, False otherwise
    """
    # Check if the file is in any excluded directory (using full paths)
    file_path_str = str(Path(file_path).resolve())
    for excluded_dir in EXCLUDE_DIRS:
        excluded_dir_resolved = str(Path(excluded_dir).resolve())
        if file_path_str.startswith(excluded_dir_resolved):
            return True
    return False


def should_skip_directory(dir_path: str | Path) -> bool:
    """
    Check if the directory should be skipped based on path.

    Args:
        dir_path: Path to the directory to check

    Returns:
        True if the directory should be skipped, False otherwise
    """
    # Check if the directory matches any excluded directory (using full paths)
    dir_path_str = str(Path(dir_path).resolve())
    for excluded_dir in EXCLUDE_DIRS:
        excluded_dir_resolved = str(Path(excluded_dir).resolve())
        if dir_path_str == excluded_dir_resolved or dir_path_str.startswith(
            excluded_dir_resolved + os.sep
        ):
            return True
    return False


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    file_pattern: str = "*.py",
) -> None:
    """
    Process all Python files in a directory and create notebooks.

    Args:
        input_dir: Directory containing Python files to process
        output_dir: Directory where notebooks will be saved
        file_pattern: Glob pattern to match Python files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Get all Python files in the input directory
    py_files = list(input_dir.glob(file_pattern))

    if not py_files:
        print(f"No {file_pattern} files found in {input_dir}")
        return

    # Filter out files that should be skipped
    files_to_process = []
    for py_file in py_files:
        # Skip __init__.py files
        if py_file.name == "__init__.py":
            continue

        # Skip files in excluded directories
        if should_skip_file(py_file):
            print(f"Skipping file in excluded directory: {py_file}")
            continue

        files_to_process.append(py_file)

    # Only create output directory if we have files to process
    if not files_to_process:
        print(f"No files to process in {input_dir} (all skipped)")
        return

    # Ensure output directory exists only when we have files to process
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(files_to_process)} files from {input_dir}")

    # Process each file
    for py_file in files_to_process:
        print(f"Processing {py_file}...")

        # Extract base name for the output file
        base_name = py_file.stem

        # Define output path
        output_path = output_dir / f"{base_name}.ipynb"

        # Create notebook
        create_notebook_from_file(py_file, output_path)


def process_directory_recursively(
    input_dir: str | Path,
    output_dir: str | Path,
    file_pattern: str = "*.py",
) -> None:
    """
    Process all Python files in a directory and its subdirectories.

    Args:
        input_dir: Root directory containing Python files to process
        output_dir: Root directory where notebooks will be saved
        file_pattern: Glob pattern to match Python files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Process the current directory
    process_directory(input_dir, output_dir, file_pattern)

    # Process subdirectories
    for subdir in [
        d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("__")
    ]:
        # Skip excluded directories using full path comparison
        if should_skip_directory(subdir):
            print(f"Skipping excluded directory: {subdir}")
            continue

        # Create corresponding output subdirectory
        subdir_output = output_dir / subdir.name
        process_directory_recursively(subdir, subdir_output, file_pattern)


def clean_output_directories() -> None:
    """
    Remove all existing notebook output directories to ensure a clean regeneration.
    """
    output_dirs = [DOCS_OUTPUT_DIR, README_OUTPUT_DIR]

    for output_dir in output_dirs:
        if output_dir.exists():
            print(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Directory does not exist (skipping): {output_dir}")


def main() -> None:
    """
    Main entry point for the script.
    """
    # Clean up existing notebooks first
    print("Cleaning up existing notebooks...")
    clean_output_directories()

    # Process each directory using top-level variables
    print("Generating new notebooks...")
    process_directory_recursively(DOCS_INPUT_DIR, DOCS_OUTPUT_DIR)
    process_directory(README_INPUT_DIR, README_OUTPUT_DIR)

    print("Notebook generation complete!")


if __name__ == "__main__":
    main()
