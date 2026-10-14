:og:description: ContextGem: Troubleshooting Issues with Small Models

.. meta::
   :description: Why small language models often fail at structured extraction with ContextGem, and how to mitigate, debug and test them.

Troubleshooting Issues with Small Models
==========================================

Small language models (e.g. 8B parameter models) often struggle with ContextGem's structured extraction tasks. This guide addresses common issues and provides practical solutions.

.. seealso::
   For general guidance on selecting appropriate models for your use case, see :doc:`Choosing the Right LLM(s) <optimization_choosing_llm>`.


⚠️ Common Issues with Small Models
------------------------------------

**"LLM did not return valid JSON" Error**
    Small models frequently fail to follow the precise JSON schema required by ContextGem's internal prompts. This manifests as:

    - ``Error when validating parsed JSON: parsed_json is None``
    - ``LLM did not return valid JSON``

**Inconsistent Results**
    Small models may produce:

    - Empty extraction results
    - Incomplete or partial extractions
    - Inconsistent formatting across multiple calls


🎯 Model Capability Requirements
---------------------------------

**Minimum Recommended Performance**
    ContextGem is tested with ``openai/gpt-6-luna`` as the baseline model. For reliable structured extraction, your model should:

    - Be able to follow detailed JSON schema instructions consistently
    - Have a sufficient context window to ingest the detailed prompt and the document content
    - Maintain attention across long prompts with complex instructions


🛠️ Mitigation Strategies for Small Models
-------------------------------------------

.. important::
   **The most effective solution is usually to upgrade to a larger, more capable model** (such as ``gpt-6-luna`` or larger). The strategies below are workarounds for situations where upgrading isn't possible.

If you must use a smaller model, try these approaches individually or in combination:

**1. Reduce Task Complexity**

.. code-block:: python

    # Extract one aspect/concept at a time instead of all at once
    results = llm.extract_all(
        document,
        max_items_per_call=1  # Analyze aspects/concepts individually
    )

**2. Limit Document Scope**

.. code-block:: python

    # Process fewer document paragraphs per call
    results = llm.extract_all(
        document,
        max_paragraphs_to_analyze_per_call=50  # Default is 0 (all paragraphs)
    )

**3. Use More Specific Aspects/Concepts**

Instead of generic aspects/concepts:

.. code-block:: python

    # ❌ Too generic - may confuse small models
    Aspect(
        name="Contract Terms",
        description="Contractual/legal details"
    )

Use targeted ones:

.. code-block:: python

    # ✅ More specific - easier for small models
    Aspect(
        name="Termination Terms",
        description="Provisions on contract termination"
    ),
    Aspect(
        name="Payment Terms",
        description="Provisions on payment schedules and amounts"
    )

**4. Choose the Right API**

For extracting document sections by topic, use **Aspects API** instead of **Concepts API**:

.. code-block:: python

    # ✅ Aspects API is designed specifically for extracting document sections by topic,
    # while Concepts API is designed for extracting/inferring specific values or entities
    # from a document or a specific section.
    from contextgem import Aspect

    project_scope = Aspect(
        name="Project Scope",
        description="Details about the scope of work"
    )

    # Paragraph references are automatically added to the extracted aspects
    results = llm.extract_aspects_from_document(document)

Instead of:

.. code-block:: python

    # ❌ Concepts API's core purpose is to extract/infer specific values or entities
    # from a document or a specific section, rather than extracting document sections
    # by topic.
    from contextgem import StringConcept

    project_scope = StringConcept(
        name="Project Scope",
        description="Details about the scope of work",
        add_references=True
    )

**5. Lower the Sampling Temperature**

ContextGem does not set a sampling temperature by default, so the default of the model applies, which for local models is often set for conversational use. A low temperature makes the output more consistent, which helps a small model to follow the required JSON structure:

.. code-block:: python

    from contextgem import DocumentLLM

    llm = DocumentLLM(
        model="ollama_chat/<model_name>",
        api_base="http://localhost:11434",
        temperature=0.2,  # the default of the model applies if not set
    )


🔍 Debugging LLM Responses
----------------------------

To see what your LLM is supposed to return, you can inspect the prompt and the model's response:

.. code-block:: python

    # Make an extraction call
    results = llm.extract_aspects_from_document(document)

    # Inspect the actual prompt sent to the LLM
    prompt = llm.get_usage()[0].usage.calls[-1].prompt
    print("Prompt sent to LLM:")
    print(prompt)

    # Check the raw response (if available)
    response = llm.get_usage()[0].usage.calls[-1].response
    print("LLM response:")
    print(response)


📊 Testing Local Models
-------------------------

Before committing to a local model for production, test it on extraction tasks in the documentation, such as:

- :ref:`Aspect Extraction from Document <quickstart:📋 aspect extraction from document>`
- :ref:`Extracting Aspect with Sub-Aspects <quickstart:🌳 extracting aspect with sub-aspects>`
- :ref:`Concept Extraction from Aspect <quickstart:🔍 concept extraction from aspect>`
- :ref:`Concept Extraction from Document <quickstart:📝 concept extraction from document (text)>`


.. important::
   **Production Applications**: For production applications, especially those requiring high accuracy (like legal document analysis), using appropriately capable models is crucial. The cost of model inference is typically far outweighed by the cost of incorrect extractions or failed processing.
