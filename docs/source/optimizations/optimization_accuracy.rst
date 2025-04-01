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


Optimizing for Accuracy
========================

When accuracy is paramount, ContextGem offers several techniques to improve extraction quality, some of which are pretty obvious:

- **🚀 Use a Capable LLM**: Choose a powerful LLM model for extraction.
- **🪄 Use Larger Segmentation Models**: Select a larger SaT model for intelligent segmentation of paragraphs or sentences, to ensure the highest segmentation accuracy in complex documents (e.g. contracts).
- **💡 Provide Examples**: For most complex concepts, add examples to guide the LLM's extraction format and style.
- **🧠 Request Justifications**: For most complex aspects/concepts, enable justifications to understand the LLM's reasoning and instruct the LLM to "think" when giving an answer.
- **📏 Limit Paragraphs Per Call**: This will reduce each prompt's length and ensure a more focused analysis.
- **🔢 Limit Aspects/Concepts Per Call**: Process a smaller number of aspects or concepts in each LLM call, preventing prompt overloading.
- **🔄 Use a Fallback LLM**: Configure a fallback LLM to retry failed extractions with a different model.


.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_accuracy.py
    :language: python
    :caption: Example of optimizing extraction for accuracy
