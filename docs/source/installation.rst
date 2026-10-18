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

.. note::
   ContextGem depends on `ONNX Runtime <https://pypi.org/project/onnxruntime/>`_, which runs the SaT models that split text into sentences. Where ONNX Runtime publishes no packages on PyPI, ContextGem does not install:

   * **Alpine Linux**, or another Linux system built on the musl C library, e.g. in the ``python:3.14-alpine`` container image. In a container, use an image based on Debian, e.g. ``python:3.14-slim``.
   * **Python 3.14 on an Intel Mac**: ONNX Runtime publishes no packages for Intel Macs from version 1.24, the first with packages for Python 3.14. On an Intel Mac, use Python 3.11-3.13.

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

    # Install the dependencies, including the development dependency groups
    uv sync --all-groups

✅ Verifying Installation
--------------------------

To verify that ContextGem is installed correctly, run in your project:

.. code-block:: bash

    uv run python -c "import contextgem; print(contextgem.__version__)"

Or, if you installed it with pip, in the environment you installed it in:

.. code-block:: bash

    python -c "import contextgem; print(contextgem.__version__)"
