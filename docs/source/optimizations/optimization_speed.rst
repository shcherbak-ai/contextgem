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

:og:description: ContextGem: Optimizing for Speed

Optimizing for Speed
=====================

For large-scale processing or time-sensitive applications, optimize your pipeline for speed:

- **🚀 Enable and Configure Concurrency**: Process multiple extractions concurrently. Adjust the async limiter to adapt to your LLM API setup.
- **📦 Use Smaller Models**: Select smaller/distilled LLMs that perform faster. (See :doc:`optimization_choosing_llm` for guidance on choosing the right model.)
- **🔄 Use a Fallback LLM**: Configure a fallback LLM to retry extractions that failed due to rate limits.
- **⚙️ Use Default Parameters**: All the extractions will be processed in as few LLM calls as possible.
- **📉 Enable Justifications Only When Necessary**: Do not use justifications for simple aspects or concepts. This will reduce the number of tokens generated.


.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_speed.py
    :language: python
    :caption: Example of optimizing extraction for speed
