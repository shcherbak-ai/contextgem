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

Use the ``@register_tool`` decorator to register a function as a tool. The tool schema is auto-generated from the function signature, type hints, and docstring.

.. literalinclude:: ../../../dev/usage_examples/docs/llm_config/chat_with_tools.py
   :language: python

**Key points:**

* Decorate functions with ``@register_tool`` to make them available as tools
* Pass decorated functions directly to ``tools=[...]`` when creating the LLM
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

Supported Type Hints
~~~~~~~~~~~~~~~~~~~~~

The schema generator supports the following Python type hints:

* **Primitives:** ``str``, ``int``, ``float``, ``bool``, ``None``
* **Collections:** ``list[T]``, ``dict[K, V]``, ``tuple[T, ...]``, ``set[T]``
* **Optionals:** ``Optional[T]``, ``T | None``
* **Unions:** ``Union[X, Y, ...]``, ``X | Y``
* **Literals:** ``Literal["a", "b", "c"]``
* **Structured:** ``TypedDict``

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


Custom Schema Override
-----------------------

For full control over the tool schema, pass an explicit OpenAI-compatible schema to ``@register_tool``:

.. code-block:: python

   @register_tool(schema={
       "type": "function",
       "function": {
           "name": "search_database",
           "description": "Search the product database",
           "parameters": {
               "type": "object",
               "properties": {
                   "query": {
                       "type": "string",
                       "description": "Search query"
                   },
                   "category": {
                       "type": "string",
                       "enum": ["electronics", "clothing", "books"],
                       "description": "Product category filter"
                   }
               },
               "required": ["query"]
           }
       }
   })
   def search_database(query: str, category: str = None) -> str:
       return f"Results for '{query}' in {category or 'all categories'}"

This is useful when you need:

* Custom parameter descriptions beyond what docstrings provide
* Specific enum values or constraints
* Complex nested schemas


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
     - List of tool functions decorated with ``@register_tool``.
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
