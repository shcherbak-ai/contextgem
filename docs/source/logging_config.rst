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

:og:description: ContextGem: Logging Configuration Guide


Logging Configuration
========================

ContextGem provides comprehensive logging to help you monitor and debug the extraction process. You can control logging behavior using environment variables.


⚙️ Environment Variables
--------------------------

ContextGem uses a single environment variable for logging configuration:

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

If you need to change logging settings after importing ContextGem, use the ``reload_logger_settings()`` function:

.. literalinclude:: ../../dev/usage_examples/docstrings/utils/reload_logger_settings.py
   :language: python
   :caption: Changing logger settings at runtime


📋 Log Format
---------------

ContextGem logs use the following format:

.. code-block:: text

   [contextgem] 2025-01-11 15:30:45.123 | INFO    | Your log message here

Each log entry includes:

* Timestamp
* Log level
* Log message
