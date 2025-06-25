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

import os
import sys

project = "ContextGem"
copyright = "2025, Shcherbak AI AS"
author = "Sergii Shcherbak"
release = "0.8.2"


# Add path to the package
sys.path.insert(0, os.path.abspath("../.."))


# Skip Pydantic internal methods and attributes from the API docs
def skip_pydantic_internals(app, what, name, obj, skip, options):
    if name.startswith("model_") or name in [
        "schema",
        "schema_json",
        "construct",
        "copy",
        "dict",
        "json",
        "validate",
        "update_forward_refs",
        "parse_obj",
        "parse_file",
        "parse_raw",
        "from_orm",
        "Config",
        "__fields__",
        "__annotations__",
        "__config__",
    ]:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_pydantic_internals)


# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinx_sitemap",
]

# Configure autosectionlabel to prefix labels with document name
# to avoid conflicts with the same section names in different documents
autosectionlabel_prefix_document = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Autodoc settings
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"  # Shows types in the parameter description
typehints_fully_qualified = True
typehints_document_rtype = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Additional settings
html_title = f"{project} {release} Documentation"
html_baseurl = "https://contextgem.dev/"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_extra_path = ["robots.txt"]
html_favicon = "_static/favicon.ico"
html_logo = "_static/favicon.ico"
html_css_files = ["custom.css"]
# Sidebar settings
html_theme_options = {
    "repository_url": "https://github.com/shcherbak-ai/contextgem",
    "repository_branch": "main",
    "path_to_docs": "docs/source/",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "show_navbar_depth": 2,
    "pygments_dark_style": "github-dark-high-contrast",
}

# Open Graph metadata
ogp_site_url = "https://contextgem.dev/"
ogp_image = "https://contextgem.dev/_static/contextgem_website_preview.png"
ogp_description = "ContextGem: Effortless LLM extraction from documents"
ogp_description_length = 200
ogp_type = "website"
ogp_use_first_image = False
html_meta = {
    "description": ogp_description,
}
