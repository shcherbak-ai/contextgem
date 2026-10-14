Supported Concepts
===================

In ContextGem, Concepts are building blocks for defining the structured data you want to extract from documents.
Each concept type is designed for different kinds of information, allowing you to build complex extraction schemas.

Available Concept Types
------------------------

ContextGem provides several types of concepts, each tailored for specific extraction needs:

- 📝 :doc:`StringConcept <string_concept>`: For extracting text values
- ✅ :doc:`BooleanConcept <boolean_concept>`: For extracting boolean (True/False) values
- 🔢 :doc:`NumericalConcept <numerical_concept>`: For extracting numerical values (integers or floats)
- 📅 :doc:`DateConcept <date_concept>`: For extracting date objects
- ⭐ :doc:`RatingConcept <rating_concept>`: For extracting numerical ratings within a defined scale
- 📊 :doc:`JsonObjectConcept <json_object_concept>`: For extracting structured data with multiple fields
- 🏷️ :doc:`LabelConcept <label_concept>`: For classification using predefined labels (multi-class or multi-label)

This section provides detailed documentation for each concept type, including usage examples and best practices.

.. note::
   All concept types are extracted with LLMs. :doc:`BooleanConcept <boolean_concept>`, :doc:`LabelConcept <label_concept>` and :doc:`RatingConcept <rating_concept>` can also be extracted with a decision model, which returns the probabilities of the possible values instead of generating text. See :doc:`../decision_models/decision_models`.
