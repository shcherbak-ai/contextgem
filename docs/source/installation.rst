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

:og:description: ContextGem: Installation

Installation
============

🔧 Prerequisites
-----------------

Before installing ContextGem, ensure you have:

* Python 3.10-3.13
* pip (Python package installer)

📦 Installation Methods
------------------------

From PyPI
~~~~~~~~~~

Using `uv <https://github.com/astral-sh/uv>`_ (recommended):

.. code-block:: bash

    uv add contextgem

Or using pip:

.. code-block:: bash

    pip install -U contextgem

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

For development, clone the repository and use uv:

.. code-block:: bash

    git clone https://github.com/shcherbak-ai/contextgem.git
    cd contextgem
    
    # Install uv if you don't have it
    pip install uv
    
    # Install dependencies including development extras
    uv sync --all-groups

✅ Verifying Installation
--------------------------

To verify that ContextGem is installed correctly, run:

.. code-block:: bash

    python -c "import contextgem; print(contextgem.__version__)"
