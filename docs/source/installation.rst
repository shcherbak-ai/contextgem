:og:description: ContextGem: Installation

.. meta::
   :description: How to install ContextGem with uv or pip, and what it requires.

Installation
============

🔧 Prerequisites
-----------------

Before installing ContextGem, ensure you have:

* Python 3.11-3.14
* `uv <https://github.com/astral-sh/uv>`_ or pip

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

    # Install uv if you don't have it:
    # https://docs.astral.sh/uv/getting-started/installation/

    # Install dependencies including development extras
    uv sync --all-groups

✅ Verifying Installation
--------------------------

To verify that ContextGem is installed correctly, run:

.. code-block:: bash

    python -c "import contextgem; print(contextgem.__version__)"
