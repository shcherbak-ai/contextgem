.. meta::
   :description: How to set the level and the output of the ContextGem logger with environment variables, change them at runtime, read the log format, and filter its Python warnings.


Logging Configuration
========================

ContextGem provides comprehensive logging to help you monitor and debug the extraction process. You can control logging behavior using environment variables. ContextGem uses a **namespaced logger** under the name ``contextgem``.


⚙️ Environment Variables
--------------------------

ContextGem uses the following environment variables for logging configuration:

**CONTEXTGEM_LOGGER_LEVEL**
  Sets the logging level. Valid values are:

  * ``TRACE`` - Most verbose, shows all log messages
  * ``DEBUG`` - Shows debug information and above
  * ``INFO`` - Shows informational messages and above (default)
  * ``SUCCESS`` - Shows success messages and above
  * ``WARNING`` - Shows warnings and errors only
  * ``ERROR`` - Shows errors and critical messages only
  * ``CRITICAL`` - Shows only critical messages
  * ``OFF`` - Completely disables logging

  **Default:** ``INFO``

**CONTEXTGEM_SUPPRESS_LITELLM_NOISE**
  ContextGem hides the non-actionable log messages of ``litellm``, at any logging level, including ``OFF``. Set the variable to ``0`` to show these messages. With any other value, they stay hidden.

  **Default:** ``1``

.. warning::
   **Not recommended:** Setting the level to ``OFF`` or above ``INFO`` (such as ``WARNING`` or ``ERROR``) may cause you to miss helpful messages, guidance, recommendations, and important information about the extraction process. The default ``INFO`` level provides a good balance of useful information without being too verbose.


🔧 Setting Environment Variables
----------------------------------

**Before importing ContextGem:**

.. code-block:: bash

   # Set logging level to WARNING
   export CONTEXTGEM_LOGGER_LEVEL=WARNING

   # Disable logging completely
   export CONTEXTGEM_LOGGER_LEVEL=OFF

**In Python before import:**

.. code-block:: python

   import os

   # Set logging level to DEBUG
   os.environ["CONTEXTGEM_LOGGER_LEVEL"] = "DEBUG"

   # Import ContextGem after setting environment variables
   import contextgem


🔄 Changing Settings at Runtime
---------------------------------

If you need to change logging settings after importing ContextGem, set the environment variables and call the ``reload_logger_settings()`` function, which applies both of them:

.. literalinclude:: ../../dev/usage_examples/docstrings/utils/reload_logger_settings.py
   :language: python
   :caption: Changing logger settings at runtime


⚠️ Warnings
-------------

ContextGem reports what needs your attention in one of two ways, by what it is about:

- **What happens while ContextGem runs is logged**, at the ``WARNING`` level or above: e.g. an extraction that fails when ``raise_exception_on_extraction_error=False`` is passed to the extraction method, a chat response cut off at the max output tokens, a request that a model or its provider refuses to answer, or the answers that a decision model refuses to give in aspect extraction. Each occurrence is logged, and ``CONTEXTGEM_LOGGER_LEVEL`` controls these messages as any other.
- **What in your code will not have the expected effect is issued as a Python warning** of the :class:`~contextgem.public.exceptions.ContextGemWarning` category, a subclass of ``UserWarning``: e.g. tools set on an LLM that is used for extraction, or an output language set together with a custom system message. Such a warning points to the line of your code that caused it, e.g. the line that creates the model, and Python shows it once for that line.

Python warnings are not log messages: ``CONTEXTGEM_LOGGER_LEVEL`` does not affect them, and they are shown also with ``OFF``. Filter them with the standard ``warnings`` module, by their category, as a filter by the module of ContextGem does not match a warning that points to your code. In your tests, you can make them errors, to catch such code before it runs in production.

.. code-block:: python

   import warnings

   from contextgem import ContextGemWarning

   # Hide all the warnings of ContextGem
   warnings.filterwarnings("ignore", category=ContextGemWarning)

   # Or only some of them, by the start of the message
   warnings.filterwarnings(
       "ignore",
       message="`output_language` applies to the extraction prompts only",
       category=ContextGemWarning,
   )

   # Or make them errors, e.g. in your tests
   warnings.filterwarnings("error", category=ContextGemWarning)


📋 Log Format
---------------

ContextGem logs use the following format:

.. code-block:: text

   [contextgem] 2025-01-11 15:30:45.123 | INFO    | Your log message here

Each log entry includes:

* Timestamp (with milliseconds)
* Log level
* Log message
