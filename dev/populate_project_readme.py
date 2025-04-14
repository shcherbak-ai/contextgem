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
This script populates the README.md file with code examples from the usage_examples/ directory.

To use it, run:

```bash
python dev/populate_project_readme.py
```
"""

README_TEMPLATE_PATH = "dev/README.TEMPLATE.md"
README_OUTPUT_PATH = "README.md"
README_FOOTER = ""


def generate_readme():
    with open(README_TEMPLATE_PATH, "r", encoding="utf-8") as template_file:
        template = template_file.read()

    # Replace markers with actual code examples
    for example_file, marker in USAGE_EXAMPLES_MAPPING.items():
        code_snippet = extract_code_from_file(example_file)
        template = template.replace(marker, code_snippet)

    with open(README_OUTPUT_PATH, "w", encoding="utf-8") as readme_file:
        readme_file.write(template)
        readme_file.write(README_FOOTER)
    print("Project README.md file populated successfully.")


def extract_code_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


# Map example files to markers in the template
USAGE_EXAMPLES_MAPPING = {
    "dev/content_snippets/feature_table.html": "{{FEATURE_TABLE}}",
    "dev/usage_examples/readme/quickstart_aspect.py": "{{QUICKSTART_ASPECT}}",
    "dev/usage_examples/readme/quickstart_concept.py": "{{QUICKSTART_CONCEPT}}",
}

if __name__ == "__main__":
    generate_readme()
