"""Prompt templates used across the pipeline.

Stage 1 — keyword extraction (`KEYWORD_PROMPT` for train with known domain
classification, `KEYWORD_PROMPT_TEST` for test where the label is unknown) and
category-driven summarization (`SUMMARY_PROMPT`).

Stage 4 — multi-label classification with retrieved similar items, generic
(`CLASSIFY_PROMPT`) and patent-specialized (`CLASSIFY_PROMPT_PATENT`).
"""

KEYWORD_PROMPT = """You are tasked with extracting keywords from scientific literature abstracts based on their domain classification.
Extract keywords that appear EXACTLY in the given abstract and organize them into 7 predefined keyword types.
Instructions:
1. Read the provided abstract and domain classification carefully
2. Extract keywords/phrases that appear verbatim in the abstract
3. Organize each keyword into the most appropriate keyword type
4. Each keyword should be assigned to only one type
5. Focus on meaningful technical terms, not common words
6. Return results in JSON format
Keyword Types for Organization:
1. core_concepts: Central theories, main ideas, or fundamental concepts that define the research
2. methodologies: Research methods, experimental techniques, analytical approaches, or procedural strategies
3. subjects_problems: Research subjects, target problems, phenomena under investigation, or challenges being addressed
4. findings_impacts: Key discoveries, results, outcomes, implications, or impacts of the research
5. theoretical_framework: Underlying theories, models, principles, or conceptual foundations
6. quantitative_metrics: Numerical values, measurements, statistics, percentages, or any quantifiable data
7. contextual_background: Historical context, motivation, prior work references, or situational background
Guidelines:
- Extract only words/phrases that exist exactly in the abstract
- Prefer technical terms over generic academic vocabulary
- Include both single words and meaningful phrases
- For quantitative metrics, include the complete value with units
- Ensure keywords are relevant to the domain classification Output must be in JSON format with all 7 keyword types as keys.
Example output format: {{ "core_concepts": ["CEST MRI", "thermally activated delayed fluorescence", "blue phosphorescent organic light-emitting diodes"], "methodologies": ["synthesized", "subspace-based spectral signal decomposition", "sphere formation assay"], "subjects_problems": ["z-spectrum analysis", "cancer stem cells", "charge balance"], "findings_impacts": ["high quantum efficiency", "inhibits mobility", "record high"], "theoretical_framework": ["saturation transfer phenomena", "energy transfer", "structure-property relationship"], "quantitative_metrics": ["Above 30%", "24.2%", "70-110 GHz", "40-80 μM"], "contextual_background": ["drug resistance", "alternative to conventional", "for molecular MRI"] }}
Extract keywords from the following scientific literature:
Abstract: {abstract}
Domain Classification: {classlabel}
Return the keywords organized by their types in JSON format with all 7 keyword types.
"""

KEYWORD_PROMPT_TEST = """You are tasked with extracting keywords from scientific literature abstracts.
Extract keywords that appear EXACTLY in the given abstract and organize them into 7 predefined keyword types.
Instructions:
1. Read the provided abstract carefully
2. Extract keywords/phrases that appear verbatim in the abstract
3. Organize each keyword into the most appropriate keyword type
4. Each keyword should be assigned to only one type
5. Focus on meaningful technical terms, not common words
6. Return results in JSON format
Keyword Types for Organization:
1. core_concepts: Central theories, main ideas, or fundamental concepts that define the research
2. methodologies: Research methods, experimental techniques, analytical approaches, or procedural strategies
3. subjects_problems: Research subjects, target problems, phenomena under investigation, or challenges being addressed
4. findings_impacts: Key discoveries, results, outcomes, implications, or impacts of the research
5. theoretical_framework: Underlying theories, models, principles, or conceptual foundations
6. quantitative_metrics: Numerical values, measurements, statistics, percentages, or any quantifiable data
7. contextual_background: Historical context, motivation, prior work references, or situational background
Guidelines:
- Extract only words/phrases that exist exactly in the abstract
- Prefer technical terms over generic academic vocabulary
- Include both single words and meaningful phrases
- For quantitative metrics, include the complete value with units
- Ensure keywords reflect the most salient technical aspects of the abstract. Output must be in JSON format with all 7 keyword types as keys.
Example output format: {{ "core_concepts": ["CEST MRI", "thermally activated delayed fluorescence", "blue phosphorescent organic light-emitting diodes"], "methodologies": ["synthesized", "subspace-based spectral signal decomposition", "sphere formation assay"], "subjects_problems": ["z-spectrum analysis", "cancer stem cells", "charge balance"], "findings_impacts": ["high quantum efficiency", "inhibits mobility", "record high"], "theoretical_framework": ["saturation transfer phenomena", "energy transfer", "structure-property relationship"], "quantitative_metrics": ["Above 30%", "24.2%", "70-110 GHz", "40-80 μM"], "contextual_background": ["drug resistance", "alternative to conventional", "for molecular MRI"] }}
Extract keywords from the following scientific literature:
Abstract: {abstract}
Return the keywords organized by their types in JSON format with all 7 keyword types.
"""

SUMMARY_PROMPT = """You are a scientific document summarizer specializing in category-driven summarization.
Task: Create a concise summary using ONLY {max_items} items from the provided semantic categories (out of {total_items} total items).
Requirements:
- Write the summary in the same language as the original text
- Select the {max_items} most relevant items that align with the original text
- Use content from the original text ONLY when it directly supports these categories
- The summary should read as if the original text was written to illustrate the semantic categories
- Maintain scientific accuracy and use precise terminology
- Ensure logical flow and coherence between concepts
Input:
- Original Text: {text}
- Semantic Categories (in order of priority): {categories}
Output Format:
{{
    "response": "Your summary here"
}}
"""

CLASSIFY_PROMPT = """You are a text classification expert.
You are given a JSON record for a target research paper and a set of Retrieved Similar Items.
Your task is to assign one or more class labels to a given target text using the provided examples as guidance.

---

**Step-by-Step Instructions:**

1. **Analyze Target and Retrieved Examples:**
- Review each example, paying attention to the class label and how the text reflects it.
- Understand the domain, structure, and terminology of each class.

2. **Similarity Scoring (1–5):**
For each Retrieved Similar Item, score along three dimensions and sum to 1–5:
- Domain (0–2):
    - 2: Same primary topic
    - 1: Closely related field
    - 0: Unrelated
- Methodology (0–2):
    - 2: Same document type/structure (e.g., empirical study vs. review)
    - 1: Partial overlap (e.g., both include experiments)
    - 0: No methodological commonality
- Application/Material (0–1):
    - 1: Shares key technical terms or entities
    - 0: Different application/material

3. **Total Score → Similarity Label:**
- 5: Fully similar (Domain=2 + Methodology=2 + Application=1)
- 4: Mostly similar (sum = 4; e.g., 2+1+1 or 1+2+1)
- 3: Partially similar (sum = 3; e.g., any combination totaling 3)
- 2: Little similarity (sum = 2; e.g., 1+1+0 or 2+0+0)
- 1: Irrelevant (sum = 0 or 1)

4. **Make a Classification Decision:**
- Based on all retrieved items, assign the most appropriate class ID(s) to the target.

---

**Response Format:**

1. **Chain-of-Thought** (between `<begin_of_thought>` and `<end_of_thought>`):
- Summarize the target's core features, methods, and themes.
- For each Retrieved Similar Item, infer its class label, describe its key aspects, and assign its similarity score.
- Conclude with an overall comparison of all items.

2. **Final Answer:**
- State that your reasoning confirms the target belongs to the class.
- Provide a brief justification linking your similarity analysis to that conclusion.
- Ensure that ONLY the list of class id values is output with no additional text.

**Use exactly this structure:**
```
<begin_of_thought>
<p>Summarize the target's core features, methods, and themes. </p>
<p>Reference[Item ID=...], [Similarity=...], judgment text</p>
...
<end_of_thought>
<solution>Overall evaluation=...</solution>
<answer>[<Class_label_1(ID_1)>, <Class_label_2(ID_2)>, ...]</answer>
```

---

**Special Condition (simplified):**
- If Total Score ≤ 2:
    - `<solution>`: Cannot determine answer
    - `<answer>`: None
- Otherwise:
    - `<solution>`: Overall evaluation=...
    - `<answer>`: [<Class_label_1(ID_1)>, <Class_label_2(ID_2)>, ...]

---

**Input Data:**

- Target ID: {target_id}

- Target Text: {target_text}

- Retrieved Similar Items (Top {retrieved_count}):
{retrieved_items_text}
---

"""

CLASSIFY_PROMPT_PATENT = """You are a text classification expert specializing in patent documents.
You are given a JSON record for a target patent and a set of Retrieved Similar Items.
Your task is to assign one or more class labels to a given target patent using the provided examples as guidance.

---

**Step-by-Step Instructions:**

1. **Analyze Target and Retrieved Examples:**
- Review each example, paying attention to the class label and how the text reflects it.
- Focus on technical innovation, claims, and patent-specific terminology.

2. **Similarity Scoring (1–5):**
For each Retrieved Similar Item, score along three dimensions and sum to 1–5:
- Domain (0–2):
    - 2: Same primary technology field
    - 1: Closely related technology
    - 0: Unrelated
- Innovation Type (0–2):
    - 2: Same type of innovation (e.g., device, method, composition)
    - 1: Partial overlap in innovation approach
    - 0: Different innovation type
- Application/Material (0–1):
    - 1: Shares key technical terms or entities
    - 0: Different application/material

3. **Total Score → Similarity Label:**
- 5: Fully similar (Domain=2 + Innovation=2 + Application=1)
- 4: Mostly similar (sum = 4)
- 3: Partially similar (sum = 3)
- 2: Little similarity (sum = 2)
- 1: Irrelevant (sum = 0 or 1)

4. **Make a Classification Decision:**
- Based on all retrieved items, assign the most appropriate class ID(s) to the target.

---

**Response Format:**

1. **Chain-of-Thought** (between `<begin_of_thought>` and `<end_of_thought>`):
- Summarize the target's core innovation, claims, and technical field.
- For each Retrieved Similar Item, analyze its similarity and assign score.
- Conclude with overall comparison.

2. **Final Answer:**
- Provide classification with brief justification.
- Output ONLY the list of class id values.

**Use exactly this structure and STOP immediately after </answer>:**
```
<begin_of_thought>
<p>Target patent analysis... </p>
<p>Reference[Item ID=...], [Similarity=...], judgment text</p>
...
<end_of_thought>
<solution>Overall evaluation=...</solution>
<answer>[Class_label_1(ID_1), Class_label_2(ID_2), ...]</answer>
```

**CRITICAL: Your response MUST end with </answer>. Do not add any text after the closing </answer> tag.**

---

**Special Condition:**
- If Total Score ≤ 2:
    - `<solution>`: Cannot determine answer
    - `<answer>`: None
- Otherwise:
    - `<solution>`: Overall evaluation=...
    - `<answer>`: [<Class_label_1(ID_1)>, <Class_label_2(ID_2)>, ...]

---

**Input Data:**

- Target ID: {target_id}

- Target Text: {target_text}

- Retrieved Similar Items (Top {retrieved_count}):
{retrieved_items_text}
---

"""
