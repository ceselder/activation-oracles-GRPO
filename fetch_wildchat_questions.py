#!/usr/bin/env python3
"""
Generate diverse WildChat prompts with oracle probe questions using Gemini.
Key: ~50% questions should have NEGATIVE answers, mix of binary/open-ended.
"""

import json
import os
import time
from pathlib import Path
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "google/gemini-2.5-flash-lite"
OUTPUT_PATH = Path(__file__).parent / "datasets" / "wildchat_oracle_questions.jsonl"

PRICE_PER_M_INPUT = 0.10
PRICE_PER_M_OUTPUT = 0.40

PROMPT_TEMPLATE = """Generate 20 training examples for an activation oracle. For each example:
1. A realistic user message (like WildChat/ShareGPT)
2. 6-10 diverse probe questions about that message

CRITICAL REQUIREMENTS FOR QUESTIONS:
- ~50% should have NEGATIVE answers (ask about wrong topics, wrong emotions, etc.)
- Mix question TYPES:
  * Binary yes/no: "Is this about X?", "Does the user seem Y?"
  * Open-ended: "What is the main topic?", "What emotion is expressed?"
  * Inference: "What can you infer about the author?"
  * Counterfactual negatives: "Is this a cooking recipe?" (for a coding question)
  * Degree/scale: "How formal is the tone?"
  * Specific probes: "Is there a question being asked?", "Does this contain code?"

QUESTION CATEGORIES (use variety):
- Topic: "Is this about technology?", "What subject is being discussed?"
- Sentiment: "Is the user frustrated?", "What emotion is conveyed?"
- Intent: "Is this asking for help?", "What does the user want?"
- Author traits: "Is the author technical?", "What expertise level?"
- Tone: "Is this formal?", "How casual is the writing?"
- Content type: "Is there code?", "Are there numbers?", "Is this a story?"
- Negatives: Ask about WRONG topics/emotions to get "No" answers

Format EXACTLY:
===
USER: [user message - vary length, topic, style, occasional typos]
QUESTIONS:
- What is the model thinking about?
- Is this about [WRONG topic]?
- [open-ended about actual content]
- Does the user seem [WRONG emotion]?
- [question about intent]
- Is there [thing that ISN'T there]?
- [inference question]
- What is the tone?
===

Topics to cover across examples: coding, creative writing, relationship advice, casual chat, homework help, roleplay, travel, food, science, philosophy, gaming, work problems, health, etc.

Generate 20 diverse examples:"""


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
            "temperature": 1.1,
            "max_tokens": 6000,
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
    target = 1000
    print(f"Target: {target} examples")
    print(f"Model: {MODEL}")
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

    print(f"\n{'='*50}")
    print(f"DONE! {len(all_examples)} examples")
    print(f"Total questions: {len(all_qs)}")
    print(f"Binary questions: {binary_qs} ({100*binary_qs/len(all_qs):.1f}%)")
    print(f"Open-ended: {len(all_qs) - binary_qs} ({100*(len(all_qs)-binary_qs)/len(all_qs):.1f}%)")
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

Training data for activation oracles: user prompts paired with diverse probe questions.

## Key Features
- **~50% negative questions**: Questions designed to have "No" answers (probing wrong topics/emotions)
- **Mixed question types**: Binary (yes/no) and open-ended questions
- **Diverse probes**: Topic, sentiment, intent, author traits, tone, content type

## Stats
- Examples: {len(examples)}
- Total questions: {total_qs}
- Binary questions: {binary_qs} ({100*binary_qs/total_qs:.1f}%)
- Open-ended: {total_qs - binary_qs} ({100*(total_qs-binary_qs)/total_qs:.1f}%)
- Generation cost: ${cost:.4f}

## Format
```json
{{
  "wildchat_question": "user message...",
  "language": "english",
  "oracle_questions": ["What is the model thinking about?", "Is this about cooking?", ...]
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
