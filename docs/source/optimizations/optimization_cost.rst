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

:og:description: ContextGem: Optimizing for Cost

Optimizing for Cost
====================

ContextGem offers several strategies to optimize for cost efficiency while maintaining extraction quality:

- **💸 Select Cost-Efficient Models**: Use smaller/distilled non-reasoning LLMs for extracting aspects and basic concepts (e.g. titles, payment amounts, dates).
- **⚙️ Use Default Parameters**: All the extractions will be processed in as few LLM calls as possible.
- **📉 Enable Justifications Only When Necessary**: Do not use justifications for simple aspects or concepts. This will reduce the number of tokens generated.
- **📊 Monitor Usage and Cost**: Track LLM calls, token consumption, and cost to identify optimization opportunities.


.. literalinclude:: ../../../dev/usage_examples/docs/optimizations/optimization_cost.py
    :language: python
    :caption: Example of optimizing extraction for cost
