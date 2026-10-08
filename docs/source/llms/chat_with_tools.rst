..
   ContextGem

   Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

:og:description: ContextGem: Chat with Tools


Chat with Tools
================

This guide explains how to use LLM tool calling (function calling) with ContextGem's chat interface.
Tools allow the LLM to invoke Python functions during a conversation to perform actions or retrieve information.

.. note::
   Tool support is only available in ``chat(...)`` and ``chat_async(...)`` methods. It is not used by extraction methods.


Basic Usage
------------

A tool is a Python function decorated with ``@register_tool``. The function defines the tool: its name is the name of the tool, its docstring describes the tool to the LLM, and its parameters, with their type hints and default values, define the parameters of the tool.

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/chat_with_tools.py
   :language: python

**Key points:**

* Decorate a function with ``@register_tool``, and pass the function itself to ``tools=[...]`` when creating the LLM, or assign the functions to ``llm.tools`` later. This is the only way to define and assign tools
* A tool function must have a docstring and type hints for all its parameters, and must not have positional-only parameters, ``*args`` or ``**kwargs``
* A parameter that has a default value, or is typed as ``Optional``, is optional for the LLM
* Tool handlers must return a string (serialize structured data with ``json.dumps`` if needed)
* Use :class:`~contextgem.public.llms.ChatSession` to maintain conversation history across turns


Docstring Formats
------------------

The schema generator extracts parameter descriptions from docstrings. Multiple formats are supported:

**Sphinx/reST**

.. code-block:: python

   @register_tool
   def my_tool(query: str) -> str:
       """
       Search for information.

       :param query: The search query
       """
       return "results"

**Google style:**

.. code-block:: python

   @register_tool
   def my_tool(query: str) -> str:
       """Search for information.

       Args:
           query: The search query
       """
       return "results"

**NumPy style:**

.. code-block:: python

   @register_tool
   def my_tool(query: str) -> str:
       """
       Search for information.

       Parameters
       ----------
       query : str
           The search query
       """
       return "results"


.. note::
   A docstring in any other format works as well: the whole docstring describes the tool,
   and the parameters are sent to the LLM without descriptions, unless they are described
   with ``Annotated`` (see :ref:`llm-tools-annotated-label`).


Schema Generation Best Practices
---------------------------------

Use TypedDict for Object Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plain ``dict`` generates a generic ``{"type": "object"}`` schema without property definitions.
Use ``TypedDict`` to specify field names and types explicitly:

.. code-block:: python

   from typing import TypedDict

   class InvoiceItem(TypedDict):
       qty: float
       price: float

   @register_tool
   def compute_total(items: list[InvoiceItem]) -> str:
       """
       Compute invoice total.

       :param items: List of invoice items
       """
       total = sum(it["qty"] * it["price"] for it in items)
       return str(total)

This generates a proper schema with typed properties:

.. code-block:: json

   {
     "items": {
       "type": "array",
       "items": {
         "type": "object",
         "properties": {
           "qty": {"type": "number"},
           "price": {"type": "number"}
         },
         "required": ["qty", "price"]
       }
     }
   }

.. note::
   If a module uses ``from __future__ import annotations``, define the types used in the type
   hints of a tool (e.g. a ``TypedDict``) at module level, not inside a function. Otherwise,
   they cannot be found when the tool schema is generated, and ``TypeError`` is raised.

Supported Type Hints
~~~~~~~~~~~~~~~~~~~~~

The schema generator supports the following Python type hints:

* **Primitives:** ``str``, ``int``, ``float``, ``bool``, ``None``
* **Collections:** ``list[T]``, ``dict[K, V]``, ``tuple[T, ...]``, ``set[T]``
* **Optionals:** ``Optional[T]``, ``T | None``
* **Unions:** ``Union[X, Y, ...]``, ``X | Y``
* **Literals:** ``Literal["a", "b", "c"]``
* **Structured:** ``TypedDict``
* **Annotated:** ``Annotated[T, ...]`` with a description and/or constraints (see :ref:`llm-tools-annotated-label`)

.. note::
   **JSON serialization of collections:** Since tool arguments are transmitted as JSON,
   ``tuple`` and ``set`` types are received as Python ``list`` at runtime (JSON only has
   arrays). If you need specific collection behavior, convert inside your function:
   ``items = set(items)`` or ``items = tuple(items)``.

**Examples:**

.. code-block:: python

   from typing import Literal, TypedDict

   class SearchFilters(TypedDict):
       category: str
       max_price: float

   @register_tool
   def search(
       query: str,
       limit: int = 10,
       sort: Literal["relevance", "date", "price"] = "relevance",
       filters: SearchFilters | None = None,
   ) -> str:
       """
       Search products.

       :param query: Search query
       :param limit: Maximum results to return
       :param sort: Sort order
       :param filters: Optional filters
       """
       return "results"


.. _llm-tools-annotated-label:

Parameter Descriptions and Constraints
---------------------------------------

Besides the docstring, a parameter can be described with ``Annotated``, next to its type. ``Annotated`` can also set constraints on the values of the parameter, with pydantic's ``Field``:

.. code-block:: python

   from typing import Annotated, Literal

   from pydantic import Field

   @register_tool
   def search_database(
       query: Annotated[str, Field(description="Search query", min_length=1)],
       category: Annotated[
           Literal["electronics", "clothing", "books"], "Product category filter"
       ] = "books",
       limit: Annotated[int, Field(description="Maximum results", ge=1, le=50)] = 10,
   ) -> str:
       """
       Search the product database.
       """
       return f"Results for '{query}' in {category}"

* A string in ``Annotated``, or ``Field(description=...)``, is the description of the parameter. It takes precedence over the description of the parameter in the docstring
* The constraints are added to the tool schema sent to the LLM, and the arguments of each tool call are validated against them. If the LLM passes an invalid value, the tool is not called: the LLM receives the validation error and can correct the call

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Constraints
     - Apply to
   * - ``gt``, ``ge``, ``lt``, ``le``, ``multiple_of``
     - ``int``, ``float``
   * - ``min_length``, ``max_length``
     - ``str``, ``list``, ``tuple``, ``set``, ``dict``
   * - ``pattern``
     - ``str``

``Annotated`` can also be used for the items of a collection (e.g. ``list[Annotated[int, Field(ge=0)]]``) and for the fields of a ``TypedDict``. A constraint that does not apply to the type of the parameter raises ``TypeError`` when the tools are set.

.. note::
   There is no way to pass a hand-written tool schema: the schema is always generated from
   the function, so that the two cannot go out of sync.


Serializing an LLM with Tools
------------------------------

An LLM with tools is serialized as usual (``to_dict()``, ``to_json()``, ``to_disk()``). The functions themselves are not stored: the serialized data holds the tool schemas generated from them. When the LLM is loaded, each tool is matched by its name to a function decorated with ``@register_tool``, so the functions of the tools have to be defined or imported, under their own names, in the code that loads the LLM.


Tool Configuration Options
---------------------------

The :class:`~contextgem.public.llms.DocumentLLM` class accepts several tool-related parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``tools``
     - ``None``
     - Functions decorated with ``@register_tool``, as a list or a tuple.
   * - ``tool_choice``
     - ``None``
     - Controls how the model uses tools. Options: ``"none"`` (model will not call tools),
       ``"auto"`` (model decides whether to call tools or respond with text),
       ``"required"`` (model must call at least one tool), or
       ``{"type": "function", "function": {"name": "..."}}`` to force a specific tool.
       When ``None``, defers to the API default (equivalent to ``"auto"`` when tools are defined).
       Note: ``"required"`` or forced function is automatically relaxed to ``"auto"`` in follow-up
       rounds, allowing the model to either call more tools or produce a final text response.
   * - ``parallel_tool_calls``
     - ``None``
     - Whether to enable parallel tool execution. When ``None``, defers to the API/model default.
       Set to ``True`` to explicitly enable or ``False`` to disable (if supported by the model).
   * - ``tool_max_rounds``
     - ``10``
     - Maximum number of tool execution rounds per request. Prevents infinite or excessively
       long tool chains.

.. note::
   **Why ``tool_choice="required"`` is relaxed to ``"auto"`` in follow-up rounds:**

   When you set ``tool_choice="required"``, the model MUST call at least one tool on the
   initial request. However, if this setting were kept for follow-up rounds (after tool
   results are returned to the model), the model would be forced to call tools again,
   potentially causing:

   - Infinite loops or hitting ``tool_max_rounds`` unnecessarily
   - Inability to produce a final text response to the user

   To solve this, ``tool_choice="required"`` (and forced function dicts like
   ``{"type": "function", "function": {"name": "..."}}``) are automatically relaxed to
   ``"auto"`` in follow-up rounds. This allows the model to either:

   - Call additional tools if needed (multi-round tool chaining)
   - Produce a final text response when done processing tool results


Return Value Requirements
--------------------------

Tool handlers must return a string. For structured data, serialize it before returning:

.. code-block:: python

   import json

   @register_tool
   def get_user_data(user_id: str) -> str:
       """
       Retrieve user data.

       :param user_id: The user ID
       """
       data = {"name": "John", "email": "john@example.com"}
       return json.dumps(data)

The LLM will interpret the returned string and incorporate it into its response.
