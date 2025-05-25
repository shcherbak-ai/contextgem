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

This section provides detailed documentation for each concept type, including usage examples and best practices.
