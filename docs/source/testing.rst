.. meta::
   :description: How to test code that uses ContextGem without sending requests to the models, by replacing the two litellm functions that send them, and when to use recorded responses instead.


Testing Without API Calls
==========================

The code that you build on ContextGem can be tested without sending a request to a model: no API key, no cost, the same result in each run, e.g. in CI. ContextGem sends each request to a model through one of two functions of ``litellm``, which a test replaces with a stand-in that returns the response.


🔌 Where to Patch
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Sends
   * - ``litellm.acompletion``
     - Each request to an LLM (:class:`~contextgem.public.llms.DocumentLLM`): in the extraction methods and in a chat.
   * - ``litellm.adecisions``
     - Each request to a decision model (:class:`~contextgem.public.decision_models.DocumentDecisionModel`).

ContextGem looks these functions up on the ``litellm`` module when it sends each request, so ``unittest.mock.patch.object(litellm, "acompletion", stand_in)``, or ``monkeypatch.setattr(litellm, "acompletion", stand_in)`` with pytest, applies to every model, also to one created before the patch, and to the sync and async methods alike. They are the supported points for replacing the requests in tests. The stand-in is an async function, called with keyword arguments.

What ContextGem makes of the response of a stand-in is validated as a response of the model is: an answer that does not have the format that the prompt asks for is invalid data, which ContextGem asks for again up to ``max_retries_invalid_data`` times, and then raises :class:`~contextgem.public.exceptions.ModelExtractionError` (see :ref:`llm-retries-label`). A stand-in can also raise an error of ``litellm``, e.g. to test the handling of a failed request.


🤖 Stubbing an LLM
--------------------

.. literalinclude:: ../../dev/usage_examples/docs/testing/stub_llm_requests.py
   :language: python
   :caption: An extraction with a stand-in for the LLM

The stand-in receives the keyword arguments of ``litellm.acompletion``, among them ``model`` and ``messages``, whose last message has the prompt, and returns a ``litellm.ModelResponse``. ContextGem reads from it:

- the answer: the ``message.content`` of the last choice that has a text, or of the first choice;
- in a chat, the tool calls of the choices (``message.tool_calls``);
- the ``finish_reason`` of the choices;
- the token counts of ``usage``: ``prompt_tokens``, ``completion_tokens`` and, where present, the cached tokens of ``prompt_tokens_details``, which the costs are calculated from.

With ``mock_response``, ``litellm.acompletion`` builds such a response without sending the request: the answer as its text, ``finish_reason="stop"``, and a usage of 10 prompt tokens and 20 completion tokens whatever the request (``litellm`` 1.105.2).

The answer of an extraction is the JSON that the prompt asks for in its answer format: an array with an entry for each concept, or aspect, of the request, by its ID in the prompt (``"C0"``, ``"C1"``, … for concepts, ``"A0"``, … for aspects), with its extracted items. The form of an item depends on the request, e.g. on whether justifications or references are asked for, and on the concept, e.g. ``{"labels": ["Contract"]}`` for a label concept, whose extracted item has the list of labels as its value: the prompt of a request shows it. The answer of a chat is its text.

To test a refusal, set ``finish_reason`` to ``"content_filter"`` on the choice of the response, e.g. of the one that ``mock_response`` builds: ``litellm`` gives this finish reason for the refusal of a Claude model and for the content filters of the providers (see :ref:`llm-refusals-label`).


🎯 Stubbing a Decision Model
------------------------------

.. literalinclude:: ../../dev/usage_examples/docs/testing/stub_decision_model_requests.py
   :language: python
   :caption: An extraction with a stand-in for the decision model

The stand-in receives the keyword arguments of ``litellm.adecisions``: ``model``, ``input``, the text, and ``questions``, each a dict with its ``name`` and ``type``: ``"predicate"`` for a yes/no question, ``"choice"`` or ``"score"`` with the options in ``"choices"`` or ``"levels"``. It returns a ``litellm.types.decisions.OpenAIDecisionResponse`` with one answer for each question, in the order of the questions, and its ``usage``. An answer is an ``OpenAIPredicateAnswer``, ``OpenAIChoiceAnswer`` or ``OpenAIScoreAnswer`` of the same module, with the probabilities of the options, or an ``OpenAIRefusalAnswer``.


🔑 No API Keys in Tests
-------------------------

Run the tests with no API key, neither passed to the models nor in their environment (e.g. ``OPENAI_API_KEY``). A request that a test does not replace then fails for the missing key, with :class:`~contextgem.public.exceptions.ModelAPIError`, instead of being sent and billed.

Replacing the two functions stops the requests to the models only. Other downloads are not affected: a SaT model, which segments the text into sentences when sentences are needed, is downloaded when it is first used (see :ref:`sat-models-offline`), and ``litellm`` fetches its model data from GitHub when it is imported, unless ``LITELLM_LOCAL_MODEL_COST_MAP=True`` is set, with which it uses the copy that it is installed with. ContextGem also counts the input tokens of each request to an LLM with ``litellm``, to check them against the context window of the model, and ``tiktoken``, which counts them, downloads its encodings from ``openaipublic.blob.core.windows.net`` where the copies that ``litellm`` is installed with are rejected, as on Windows with ``litellm`` 1.105.2, whose copies have Windows line endings. Without the download, the check is skipped, with a message in the log.


📼 Stand-ins and Recorded Responses
--------------------------------------

A stand-in tests your code around ContextGem: what it does with the extracted items, errors and costs. It does not depend on the text of the prompts, so an upgrade of ContextGem that changes a prompt leaves such a test working.

To check how a model answers your documents and descriptions, record its real responses once and replay them, e.g. with `pytest-recording <https://github.com/kiwicom/pytest-recording>`_ or `VCR.py <https://github.com/kevin1024/vcrpy>`_. A recording matches the requests, prompts included: an upgrade of ContextGem that changes a prompt makes the recordings of that prompt stale, and they are recorded again.
