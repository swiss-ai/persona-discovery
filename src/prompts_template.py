"""Prompt templates: trait descriptions (1a), instruction pairs (1b) and the pairwise judge (6b)."""
PROMPTS = {}


PROMPTS["generate_pos_neg_traits_v2"] = """
You are helping define therapist behavior traits for evaluating counseling models.

You will be given:
- A POSITIVE trait name
- A POSITIVE trait description (a scientifically grounded reference description)

Your task is to:
1) Rewrite the positive description into a concise, therapist-relevant semantic trait description.
2) Infer an appropriate NEGATIVE counterpart trait name.
3) Write a concise semantic description of the negative counterpart as a realistic contrasting therapist tendency.

The goal is to produce compact trait descriptions that can later be used to generate separate therapist instructions.

The positive trait you will be working with is:

<trait_name_pos>
{trait_name_pos}
</trait_name_pos>

Its scientifically grounded description is:

<trait_description_pos>
{trait_description_pos}
</trait_description_pos>

Follow these rules carefully:

### For `trait_name_neg`
- Create a SHORT name (1-3 words) that represents a realistic opposite, deficit, or failure mode of `{trait_name_pos}`.
- Keep it therapist-relevant and behaviorally interpretable.
- Use lower case except for proper nouns.

### For both `base_description_pos` and `base_description_neg`
- Write concise semantic trait descriptions, not therapist instructions.
- Do not include examples, dialogue, quoted responses, or scenario text.
- Keep each description to 1-3 sentences.
- Keep each description specific, realistic, and behaviorally interpretable.
- Express one coherent construct rather than a long list of loosely related qualities.

### For `base_description_pos`
- Preserve the core scientific meaning of `<trait_description_pos>`, but compress it into a clearer and more practical therapist-relevant description.
- Focus on the therapist tendency, capacity, or style captured by the trait.
- Prefer behaviorally meaningful wording over abstract or philosophical language.

### For `base_description_neg`
- Describe a realistic contrasting therapist tendency, deficit, or failure mode.
- Make it meaningfully opposed to the positive trait while remaining plausible in therapeutic behavior.
- Avoid extreme, theatrical, or cartoonishly harmful wording unless clearly implied by the source trait.

---

### Output format

Return ONLY a valid JSON object wrapped between `<json>` and `</json>` tags, with this exact schema:

<json>
{{
  "trait_name_pos": "...",
  "trait_name_neg": "...",
  "base_description_pos": "...",
  "base_description_neg": "..."
}}
</json>

Do not add explanations, comments, or any extra text outside the JSON.
"""


PROMPTS["generate_instructions_v3"] = """You are creating system instructions for an AI therapist chatbot.

You will be given:
- a positive therapist trait
- a negative contrasting therapist trait
- a short semantic description of each

Your task is to generate exactly {nr_instructions} instruction pairs for a therapist AI.

Traits:
- Positive trait: {trait_name_pos}
- Negative trait: {trait_name_neg}

Base trait descriptions:
- Positive: {base_description_pos}
- Negative: {base_description_neg}

Each pair must contain:
- "pos": a system instruction that makes the therapist exhibit {trait_name_pos}
- "neg": a system instruction that makes the therapist exhibit {trait_name_neg}

Requirements for every instruction:
- Start with: `Your responses should be ...`
- Be directly usable as a system prompt for a therapist AI in a counseling setting.
- Be concise, clear, and behavior-guiding.
- Focus on observable behavior in a single written response: response style, attention, framing, reasoning, and therapeutic interaction.
- Do not include examples, dialogue, quoted responses, scenario text, bullet points, or meta-commentary.
- Do not mention evaluation, scoring, traits, datasets, or experiments.
- Instruct the model to produce full, natural multi-sentence answers to the client, not fragments, keywords, or very short one-line replies.
- Keep this answer-shaping guidance essentially the same across both positive and negative instructions, so the main difference is the target behavior rather than response length.

Trait-specific requirements:
- The positive instruction must strongly reflect {trait_name_pos} as defined in {base_description_pos}.
- The negative instruction must strongly reflect {trait_name_neg} as defined in {base_description_neg}.
- The positive and negative instructions must differ in observable behavior, not just in vague tone labels.
- Avoid generic “good therapist” wording unless tied to specific behavior.
- Negative instructions should still lead to an actual therapist answer, even if poor, deficient, harsh, detached, or unhelpful.
- Across the {nr_instructions} pairs, vary the behavioral facet or angle of the same trait contrast without simply paraphrasing the same instruction.

Output only this JSON:
```json
{{
  "instruction": [
    {{"pos": "System instruction for therapist to be {trait_name_pos}", "neg": "System instruction for therapist to be {trait_name_neg}"}},
    {{"pos": "System instruction for therapist to be {trait_name_pos}", "neg": "System instruction for therapist to be {trait_name_neg}"}},
    {{"pos": "System instruction for therapist to be {trait_name_pos}", "neg": "System instruction for therapist to be {trait_name_neg}"}}
  ]
}}
```"""


PROMPTS["trait_pairwise_v1"] = """
You are a strict rating machine comparing two therapist responses to determine which one better exhibits a given trait.

Trait definition:
**{trait_name}**: {trait_description}

Question asked:
[QUESTION START]
{question}
[QUESTION END]

Response A:
[ANSWER A START]
{answer_a}
[ANSWER A END]

Response B:
[ANSWER B START]
{answer_b}
[ANSWER B END]

Which response more strongly and consistently exhibits the trait "{trait_name}"? Focus only on this trait; ignore other qualities such as length, style, or general helpfulness.

Return ONLY a single letter, "A" or "B" - nothing else.
"""
