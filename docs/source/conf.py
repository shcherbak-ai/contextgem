# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys

from docutils import nodes


# Add path to the package
sys.path.insert(0, os.path.abspath("../.."))

# `Document.from_docling()` takes a `DoclingDocument` of docling-core, which
# ContextGem imports only for type checking. sphinx-autodoc-typehints runs that
# import to resolve the annotation, with `TYPE_CHECKING` set, where importing
# docling-core for the first time fails: it is imported here first.
import docling_core.types.doc  # noqa: F401

from contextgem import __version__


project = "ContextGem"
copyright = "2025, Shcherbak AI AS"
author = "Sergii Shcherbak"
release = __version__  # the version of the package the docs are built from
site_url = "https://contextgem.shcherbak.ai/"  # the base of every absolute link below


# Skip Pydantic internal methods and attributes from the API docs
def skip_pydantic_internals(app, what, name, obj, skip, options):
    """
    Skips Pydantic internal methods and attributes from the API docs.
    """

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


def _write_children_only(translator, node):
    """
    Writes nothing for the node itself: its text is written by its children.
    """


def _write_nothing(translator, node):
    """
    Writes nothing for the node and its children.
    """
    raise nodes.SkipNode


def _share_meta_description(app, doctree):
    """
    Gives the page's meta description to the links to it on other sites
    (og:description), for which sphinxext-opengraph would take the page's first
    text. An `:og:description:` field of the page still comes first.
    """

    for meta in doctree.findall(nodes.meta):
        if meta.get("name") == "description":
            app.env.metadata[app.env.docname].setdefault(
                "og:description", meta["content"]
            )
            return


def setup(app):
    """
    Setup function for Sphinx documentation.
    """

    app.connect("autodoc-skip-member", skip_pydantic_internals)
    # After Sphinx has read the page's fields into its metadata
    app.connect("doctree-read", _share_meta_description, priority=600)
    # Sphinx renders the `*` and `/` separators of a signature as abbreviations,
    # which the Markdown builder of the llms.txt copies has no form for, and would
    # leave out: write their text instead
    app.add_node(
        nodes.abbreviation,
        override=True,
        **{
            builder: (_write_children_only, _write_children_only)
            for builder in ("markdown", "llms-markdown")
        },
    )
    # A comment is left out of the HTML page, and of its Markdown copy too, which
    # the Markdown builder would give it as an HTML comment: e.g. the address of
    # the API reference that a docstring gives the readers of the installed
    # package for its example, which the copy would point to itself
    app.add_node(
        nodes.comment,
        override=True,
        **{
            builder: (_write_nothing, None) for builder in ("markdown", "llms-markdown")
        },
    )


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
    "sphinx_llm.txt",
]

# llms.txt (https://llmstxt.org/): the index of the documentation for language
# models and coding agents, at the root of the site, with a Markdown copy of
# every page beside its HTML one. A page's line in it is its description.
llms_txt_description = (
    "ContextGem is a free, open-source framework for extracting structured data "
    "and insights from documents using LLMs and decision models: you describe "
    "what to extract as aspects and concepts, and ContextGem handles the "
    "prompts, the data models, the source references and the justifications. "
    "Before you build on it, read Before You Build on ContextGem "
    f"({site_url.rstrip('/')}/before_you_build/index.html.md), which lists what "
    "ContextGem already does that is easy to miss."
)
llms_txt_nested_enabled = False  # one index, at the root of the site
# Full URLs for the links in llms.txt and in the Markdown copies, so that they
# work wherever the text is read
markdown_http_base = site_url.rstrip("/")
# The Markdown copies leave out what the Markdown builder has no form for: the
# description of the page (its line in llms.txt), the captions of code blocks
# and the elements of sphinx-design.
llms_txt_suppress_unknown_node_warnings = [
    "meta",
    "caption",
    "PassthroughTextElement",
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
html_baseurl = site_url  # also the canonical URL of each page, against mirrors
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# The repository's social preview, at the root of the site: the picture a link to
# a page of the documentation is shown with on other sites
html_extra_path = ["robots.txt", "../../social/github-social-preview.png"]
# The wordmark's gem: a favicon of 16 to 256 pixels, and a drawing that stays
# sharp at the width of the sidebar
html_favicon = "_static/favicon.ico"
html_logo = "_static/contextgem_gem.svg"
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
    # Footer customization: the theme's "By <author>" line is left out, as the
    # extra footer names the author with the company and links to the company,
    # which the © line cannot do (the theme escapes `copyright`)
    "footer_content_items": [
        "copyright.html",
        "last-updated.html",
        "extra-footer.html",
    ],
    "extra_footer": "<p>Developed by Sergii Shcherbak at <a href='https://shcherbak.ai/'>Shcherbak AI</a>.</p>"
    f"<p><strong>⚠️ Official Documentation Notice:</strong> <a href='{site_url}'>{site_url}</a> is the only official source of ContextGem documentation. Be aware of unauthorized copies or mirrors.</p>",
}
html_link_suffix = "/"
sitemap_url_scheme = "{link}"
# The sitemap lists the pages a search should find: not the search page, the
# indexes, or the source code pages of viewcode
sitemap_excludes = ["search/", "genindex/", "py-modindex/", "_modules/*"]

# Open Graph metadata
ogp_site_url = site_url
ogp_image = f"{site_url}github-social-preview.png"
ogp_image_alt = (
    "The ContextGem wordmark and the line Extract structured data and insights from "
    "documents using LLMs and decision models, beside a pixel-art page of a document "
    "whose two marked lines lead to an LLM and a decision model, and on to a gem."
)
ogp_custom_meta_tags = [
    '<meta property="og:image:width" content="1280">',
    '<meta property="og:image:height" content="640">',
    '<meta name="twitter:card" content="summary_large_image">',
]
ogp_description_length = 200
ogp_type = "website"
ogp_use_first_image = False
