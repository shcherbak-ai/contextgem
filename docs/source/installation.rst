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
~~~~~~~~~

The simplest way to install ContextGem is via pip:

.. code-block:: bash

    pip install -U contextgem

From Source
~~~~~~~~~~~

To install from source:

.. code-block:: bash

    git clone https://github.com/shcherbak-ai/contextgem.git
    cd contextgem
    pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development, we use Poetry:

.. code-block:: bash

    # Install poetry if you don't have it
    pip install poetry
    
    # Install dependencies including development extras
    poetry install --with dev
    
    # Activate the virtual environment
    poetry shell

✅ Verifying Installation
--------------------------

To verify that ContextGem is installed correctly, run:

.. code-block:: bash

    python -c "import contextgem; print(contextgem.__version__)"