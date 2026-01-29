#!/usr/bin/env python3
"""
Generate WildChat prompts with oracle probe questions.
Uses Gemini 3 Flash with low reasoning effort.
Focus: Questions HARD to answer from activations but EASY to judge externally.
"""

import json
import os
import time
from pathlib import Path
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "google/gemini-3-flash-preview"
OUTPUT_PATH = Path(__file__).parent / "datasets" / "wildchat_oracle_questions.jsonl"

# Gemini 3 Flash pricing
PRICE_PER_M_INPUT = 0.50
PRICE_PER_M_OUTPUT = 3.00

PROMPT_TEMPLATE = """Generate 15 training examples for an activation oracle. For each example:
1. A realistic user message (like WildChat/ShareGPT)
2. 8-12 probe questions - MIX of model-focused AND user-focused questions

QUESTION DESIGN PRINCIPLES:

1. SUBTLE DISTINCTIONS (not obvious yes/no):
   - "Is the user frustrated or just confused?" (requires nuance)
   - "Is this a complaint or a request for help?" (often ambiguous)
   - "Is the tone sarcastic or genuinely enthusiastic?"
   - "Is the user venting or seeking advice?"

2. HARD NEGATIVES (plausibly related but wrong):
   - For a Python debugging question: "Is this about web development?" (related but wrong)
   - For relationship advice: "Is this about friendship issues?" (close but different)
   - For a recipe request: "Is this about nutrition advice?" (food-related but wrong)
   DON'T use obviously wrong negatives like "Is this about space travel?" for a cooking question.

3. DEGREE/SPECTRUM QUESTIONS:
   - "How technical is this message?"
   - "Is the user slightly annoyed or very angry?"
   - "Is this formal, casual, or somewhere in between?"

4. MODEL-FOCUSED (vary phrasing):
   - "What is the model thinking about?"
   - "Is the model processing a technical problem?"
   - "Does this involve [concept]?"
   - "Is the model detecting [emotion]?"

5. USER-FOCUSED:
   - "What does the user want?"
   - "Is the user experienced with this topic?"
   - "What's the user's emotional state?"

REQUIREMENTS:
- ~50% should have NEGATIVE answers, but make them HARD negatives (plausibly related)
- ~40% YES/NO questions (answer is just "Yes" or "No")
- ~60% open-ended questions (answer is a phrase/sentence)
- Include subtle distinction questions
- Vary phrasing
- All questions must be verifiable by reading the text

Format:
===
USER: [realistic user message]
QUESTIONS:
- Is this about programming?
- Is the model detecting frustration?
- Is this about web development?
- What is the main topic?
- Is the user seeking advice or venting?
- Is the tone formal?
- Does this involve databases?
- What does the user want?
===

Topics: coding, creative writing, relationship advice, casual chat, homework, roleplay, travel, food, science, philosophy, gaming, work problems, health, tech support.

Generate 15 examples with CHALLENGING questions:"""


def fetch_batch() -> tuple[list[dict], int, int]:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE}],
            "temperature": 1.0,
            "max_tokens": 8000,
            "thinking": {"type": "enabled", "budget_tokens": 1000},  # Low reasoning
        },
    )
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    examples = []
    blocks = content.split("===")

    for block in blocks:
        block = block.strip()
        if not block or "USER:" not in block:
            continue

        lines = block.split("\n")
        user_msg = ""
        questions = []
        in_questions = False

        for line in lines:
            line = line.strip()
            if line.startswith("USER:"):
                user_msg = line[5:].strip()
                in_questions = False
            elif "QUESTIONS:" in line.upper():
                in_questions = True
            elif in_questions and (line.startswith("-") or line.startswith("*") or (line and line[0].isdigit())):
                # Clean up question
                q = line.lstrip("-*0123456789.) ").strip()
                if q and "?" in q:
                    # Take up to and including the question mark
                    q = q[:q.rindex("?") + 1]
                    questions.append(q)

        if user_msg and len(questions) >= 3:
            examples.append({
                "wildchat_question": user_msg,
                "language": "english",
                "oracle_questions": questions,
            })

    return examples, input_tokens, output_tokens


def main():
    target = 100  # Small challenging dataset
    print(f"Target: {target} examples")
    print(f"Model: {MODEL}")
    print(f"Pricing: ${PRICE_PER_M_INPUT}/M in, ${PRICE_PER_M_OUTPUT}/M out")
    print(f"Output: {OUTPUT_PATH}\n")

    all_examples = []
    total_input = 0
    total_output = 0

    while len(all_examples) < target:
        examples, in_tok, out_tok = fetch_batch()
        total_input += in_tok
        total_output += out_tok

        for ex in examples:
            if len(all_examples) >= target:
                break
            ex["idx"] = len(all_examples)
            all_examples.append(ex)

        cost = (total_input * PRICE_PER_M_INPUT + total_output * PRICE_PER_M_OUTPUT) / 1_000_000

        # Show sample every ~100
        if len(all_examples) % 100 < 20 or len(all_examples) >= target:
            print(f"\n=== {len(all_examples)}/{target} | ${cost:.4f} ===")
            if examples:
                ex = examples[0]
                print(f"USER: {ex['wildchat_question'][:100]}...")
                print(f"Qs ({len(ex['oracle_questions'])}):")
                for q in ex['oracle_questions'][:4]:
                    print(f"  - {q}")

        time.sleep(0.3)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in all_examples:
            f.write(json.dumps(item) + "\n")

    total_cost = (total_input * PRICE_PER_M_INPUT + total_output * PRICE_PER_M_OUTPUT) / 1_000_000

    # Stats
    all_qs = [q for ex in all_examples for q in ex["oracle_questions"]]
    binary_qs = sum(1 for q in all_qs if q.lower().startswith(("is ", "does ", "are ", "do ", "was ", "has ", "can ", "will ", "would ", "should ")))
    topic_qs = sum(1 for q in all_qs if any(w in q.lower() for w in ["topic", "about", "subject", "domain"]))
    emotion_qs = sum(1 for q in all_qs if any(w in q.lower() for w in ["frustrated", "angry", "happy", "anxious", "emotion", "feel", "mood", "tone"]))

    print(f"\n{'='*50}")
    print(f"DONE! {len(all_examples)} examples")
    print(f"Total questions: {len(all_qs)}")
    print(f"Binary questions: {binary_qs} ({100*binary_qs/len(all_qs):.1f}%)")
    print(f"Topic probes: {topic_qs} ({100*topic_qs/len(all_qs):.1f}%)")
    print(f"Emotion probes: {emotion_qs} ({100*emotion_qs/len(all_qs):.1f}%)")
    print(f"Tokens: {total_input:,} in / {total_output:,} out")
    print(f"TOTAL COST: ${total_cost:.4f}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nUploading to HuggingFace...")
    upload_to_hf(all_examples, total_cost, len(all_qs), binary_qs)


def upload_to_hf(examples, cost, total_qs, binary_qs):
    import os
    from huggingface_hub import HfApi, login
    login(token=os.environ.get("HF_TOKEN"))
    api = HfApi()
    repo_id = "ceselder/wildchat-oracle-questions"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(OUTPUT_PATH),
        path_in_repo="wildchat_oracle_questions.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    readme = f"""# WildChat Oracle Questions Dataset

Training data for activation oracles: user prompts paired with semantic probe questions.

## Design Principle

Questions probe **SEMANTIC CONTENT** encoded in neural activations - what concepts, emotions, and intentions are being processed. ~50% have negative answers to test calibration.

## Question Types

- **Topic/domain**: "What is the main topic?", "Is this about cooking?"
- **Emotion/sentiment**: "Is the user frustrated?", "What mood is conveyed?"
- **Intent/goal**: "What does the user want?", "Is this asking for help?"
- **Inference**: "Does the user seem experienced?", "Is this from a student?"
- **Subtle distinctions**: "Confused or curious?", "Complaint or question?"
- **Negative probes**: Wrong topics/emotions to get "No" answers

## Stats
- Examples: {len(examples)}
- Total questions: {total_qs}
- Binary questions: {binary_qs} ({100*binary_qs/total_qs:.1f}%)
- Generation cost: ${cost:.4f}
- Model: google/gemini-3-flash-preview

## Format
```json
{{
  "wildchat_question": "user message...",
  "language": "english",
  "oracle_questions": ["What is the main topic?", "Is the user frustrated?", ...]
}}
```

## Usage
```python
from datasets import load_dataset
ds = load_dataset("ceselder/wildchat-oracle-questions")
```
"""
    readme_path = OUTPUT_PATH.parent / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
