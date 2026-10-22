.. meta::
   :description: Why and how to keep the dependencies of an application that uses ContextGem under control: a lockfile, new dependency releases only once they are a week old, and checks against known advisories.


Dependency Security
====================

Like any open-source Python library, ContextGem builds on other open-source packages: an install brings in dozens of them, from LiteLLM and Pydantic to the packages that those use in turn, and more with the ``docling`` extra. ContextGem gives a range of versions for most of them, so that it installs alongside the other packages of your project, and within those ranges an install takes the newest releases.

Packages on PyPI are a target of supply-chain attacks, in which an attacker who gains a maintainer's access uploads a release with malicious code. LiteLLM, which ContextGem uses to call the model providers, had two such attacks: versions 1.82.7 and 1.82.8 in March 2026, and 1.107.0 in October 2026 (`MAL-2026-2144 <https://osv.dev/vulnerability/MAL-2026-2144>`_), all three removed from PyPI since. ContextGem pins LiteLLM to the versions that its tests run on, and none of its releases required any of these. The practices below keep what your application installs under your control.


🔒 Install from a Lockfile
-----------------------------

With uv, ``uv add contextgem`` records each version that it installs in ``uv.lock``, with the hashes of its files. Commit the lockfile, and install from it where the application runs:

.. code-block:: bash

    uv sync --locked

Each install then gets the same versions, checked against their hashes, until you upgrade them with ``uv lock --upgrade``.


⏳ Take New Releases a Week Later
-----------------------------------

A malicious release is usually found and removed within days of its upload, so a cutoff that takes each new release only once it is a week old keeps it out of your lockfile. Set the cutoff in the ``pyproject.toml`` of your project (see `exclude-newer <https://docs.astral.sh/uv/reference/settings/#exclude-newer>`_):

.. code-block:: toml

    [tool.uv]
    exclude-newer = "7 days"

``uv add`` and ``uv lock`` then take the newest versions that are at least a week old. To require a newer release of a package, name its version, e.g. ``uv add "contextgem>=1.2"``: if the cutoff keeps out that release, or a version that it requires, uv names the package, which you can take early, as below, or once it is a week old.


🩹 Take a Security Fix Early
-------------------------------

A security fix is often younger than a week. To take it before the cutoff admits it, give its package a cutoff of its own in ``pyproject.toml``, the time of the fix's upload (see `exclude-newer-package <https://docs.astral.sh/uv/reference/settings/#exclude-newer-package>`_):

.. code-block:: toml

    [tool.uv.exclude-newer-package]
    example-package = "2026-10-20T12:00:00Z"

Remove the entry once the release is a week old, so that the package follows the cutoff of your project again.


🔍 Check for Known Advisories
--------------------------------

`osv.dev <https://osv.dev>`_ collects the advisories of PyPI packages, from known vulnerabilities to reports of malicious releases, such as the one above. uv checks the versions of your lockfile against it (see `uv audit <https://docs.astral.sh/uv/reference/cli/#uv-audit>`_, which uv marks as experimental):

.. code-block:: bash

    uv audit

An advisory can come out after a version is locked, so run the check regularly, e.g. in CI, and upgrade past what it reports.


📌 What ContextGem Does
--------------------------

* ContextGem pins a few dependencies, e.g. LiteLLM, to the versions that its tests run on. It moves a pin to a new version only once its tests pass with it, so that a change in the API of a dependency does not break ContextGem in your application.
* When a dependency releases a security fix, ContextGem allows it in its version ranges, and raises its minimum version where no other package needs an older one, e.g. for the root certificates of certifi.
* Its development follows the practices above: its lockfile takes new releases once they are a week old, and each locked version is checked against the advisories of osv.dev at each commit.
* Each release is built and published on GitHub Actions, with an attestation of the workflow that built it, which PyPI shows on the page of each file.
