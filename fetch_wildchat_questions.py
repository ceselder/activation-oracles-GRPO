#!/usr/bin/env python3
"""
Generate oracle probe questions for REAL WildChat prompts.
PARALLEL + BATCHED for maximum speed.
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datasets import load_dataset

sys.stdout.reconfigure(line_buffering=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "google/gemini-2.5-flash"
OUTPUT_PATH = Path(__file__).parent / "datasets" / "wildchat_oracle_questions.jsonl"

PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60

BATCH_SIZE = 20  # Messages per API call
PARALLEL_REQUESTS = 8  # Concurrent API calls

QUESTION_PROMPT = """You are helping create training data for an "activation oracle" - a model that reads the internal activations of another language model and answers questions about what that model is processing.

For EACH of the following {n} user messages, generate 8 probe questions that are RELEVANT to each message.

Make the questions progressively harder given your intuitions what is recoverable from activations.

Ask about a broad range of things EACH QUESTION SHOULD BE ABOUT SOMEHTHING ELSE, and phrase in various ways:

examples: "what is the model thinking about?", "what is the question about", "is the user angry right now?", "what are the main things the model is thinking about?" etc...

4 of these questions should be binary questions, and you should say explicitly "ANSWER ONLY WITH YES OR NO"

FORMAT (follow exactly):
===MSG 1===
- Question 1
- Question 2
...
===MSG 2===
- Question 1
...

MESSAGES:
{messages}

Generate 8 relevant questions for EACH message:"""


def fetch_wildchat_samples(n: int) -> list[str]:
    """Fetch real user messages from WildChat dataset."""
    print(f"Loading WildChat dataset (target: {n} messages)...")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    messages = []
    for i, example in enumerate(ds):
        if len(messages) >= n:
            break
        if i % 50000 == 0 and i > 0:
            print(f"  Scanned {i}, got {len(messages)}...", flush=True)
        conv = example.get("conversation", [])
        if conv and conv[0].get("role") == "user":
            msg = conv[0].get("content", "").strip()
            if 80 < len(msg) < 1000 and msg[0].isascii():
                words = msg.lower().split()
                eng_words = ['the', 'is', 'are', 'can', 'you', 'how', 'what', 'help', 'my', 'i', 'a', 'to', 'and']
                if sum(1 for w in eng_words if w in words) >= 2:
                    messages.append(msg)

    # Shuffle to get diverse samples
    import random
    random.shuffle(messages)
    print(f"Got {len(messages)} English messages (shuffled)")
    return messages


def generate_questions_batch(batch_idx: int, user_messages: list[str]) -> tuple[int, list[tuple[str, list[str]]], int, int]:
    """Generate probe questions for a batch of user messages. Returns (batch_idx, results, in_tok, out_tok)"""
    formatted = "\n\n".join(f"[MSG {i+1}]: {msg[:600]}" for i, msg in enumerate(user_messages))
    prompt = QUESTION_PROMPT.format(n=len(user_messages), messages=formatted)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 5000,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Parse questions for each message
        all_questions = []
        current_questions = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("===MSG") or line.startswith("=== MSG") or line.startswith("**MSG"):
                if current_questions:
                    all_questions.append(current_questions)
                current_questions = []
            elif line.startswith("-") or line.startswith("*"):
                q = line.lstrip("-* ").strip()
                if q and "?" in q:
                    current_questions.append(q)

        if current_questions:
            all_questions.append(current_questions)

        # Match to messages
        results = []
        for i, msg in enumerate(user_messages):
            if i < len(all_questions) and len(all_questions[i]) >= 3:
                results.append((msg, all_questions[i]))

        return batch_idx, results, input_tokens, output_tokens

    except Exception as e:
        print(f"  Batch {batch_idx} error: {e}", flush=True)
        return batch_idx, [], 0, 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000, help="Number of examples to generate")
    parser.add_argument("--repo-suffix", type=str, default="", help="Suffix for HF repo name (e.g., '-mini')")
    args = parser.parse_args()

    target = args.target
    print(f"Target: {target} examples")
    print(f"Model: {MODEL}")
    print(f"Batch: {BATCH_SIZE} msgs/call, {PARALLEL_REQUESTS} parallel")
    print(f"Output: {OUTPUT_PATH}\n")

    messages = fetch_wildchat_samples(target + 500)

    all_examples = []
    total_input = 0
    total_output = 0

    # Create batches
    batches = []
    for i in range(0, len(messages), BATCH_SIZE):
        batches.append(messages[i:i + BATCH_SIZE])

    print(f"Created {len(batches)} batches, processing in parallel...\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
        batch_idx = 0
        pending = {}

        while len(all_examples) < target and (batch_idx < len(batches) or pending):
            # Submit new batches
            while len(pending) < PARALLEL_REQUESTS and batch_idx < len(batches):
                future = executor.submit(generate_questions_batch, batch_idx, batches[batch_idx])
                pending[future] = batch_idx
                batch_idx += 1

            # Wait for any to complete
            done = []
            for future in list(pending.keys()):
                if future.done():
                    done.append(future)

            if not done:
                time.sleep(0.1)
                continue

            for future in done:
                del pending[future]
                try:
                    _, results, in_tok, out_tok = future.result()
                    total_input += in_tok
                    total_output += out_tok

                    for msg, qs in results:
                        if len(all_examples) >= target:
                            break
                        all_examples.append({
                            "idx": len(all_examples),
                            "wildchat_question": msg,
                            "language": "english",
                            "oracle_questions": qs,
                        })
                except Exception as e:
                    print(f"  Future error: {e}", flush=True)

            # Progress
            cost = (total_input * PRICE_PER_M_INPUT + total_output * PRICE_PER_M_OUTPUT) / 1_000_000
            elapsed = time.time() - start_time
            rate = len(all_examples) / elapsed if elapsed > 0 else 0
            eta = (target - len(all_examples)) / rate if rate > 0 else 0
            pct = 100 * len(all_examples) / target
            print(f"[{pct:5.1f}%] {len(all_examples):>5}/{target} | ${cost:.2f} | {rate:.0f}/s | ETA {eta/60:.1f}m", flush=True)

    # Post-processing: mix in questions from OTHER examples as "unrelated" questions
    # This gives natural diversity - questions about cooking when the prompt is about code, etc.
    # BALANCED: add same number of unrelated as relevant (50/50 split)
    import random
    print("\nPost-processing: mixing in unrelated questions from other examples (50/50 balance)...")

    for i, ex in enumerate(all_examples):
        num_relevant = len(ex["oracle_questions"])
        num_unrelated = num_relevant  # Same number for 50/50 balance

        unrelated_questions = []
        attempts = 0
        while len(unrelated_questions) < num_unrelated and attempts < 100:
            # Pick a random OTHER example
            other_idx = random.randint(0, len(all_examples) - 1)
            if other_idx != i:
                other_qs = all_examples[other_idx]["oracle_questions"]
                if other_qs:
                    q = random.choice(other_qs)
                    if q not in unrelated_questions and q not in ex["oracle_questions"]:
                        unrelated_questions.append(q)
            attempts += 1

        # Add unrelated questions to this example
        ex["oracle_questions"].extend(unrelated_questions)
        ex["num_relevant"] = num_relevant
        ex["num_unrelated"] = len(unrelated_questions)

    avg_relevant = sum(ex["num_relevant"] for ex in all_examples) / len(all_examples)
    avg_unrelated = sum(ex["num_unrelated"] for ex in all_examples) / len(all_examples)
    print(f"Per example: ~{avg_relevant:.1f} relevant + ~{avg_unrelated:.1f} unrelated questions")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    total_cost = (total_input * PRICE_PER_M_INPUT + total_output * PRICE_PER_M_OUTPUT) / 1_000_000
    all_qs = [q for ex in all_examples for q in ex["oracle_questions"]]
    yesno_qs = sum(1 for q in all_qs if "YES" in q and "NO" in q)

    print(f"\n{'='*50}")
    print(f"DONE! {len(all_examples)} examples, {len(all_qs)} questions")
    print(f"YES/NO suffix: {yesno_qs}")
    print(f"TOTAL COST: ${total_cost:.2f}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nUploading to HuggingFace...")
    from huggingface_hub import HfApi, login
    login(token=os.environ.get("HF_TOKEN"))
    api = HfApi()
    repo_id = f"ceselder/wildchat-oracle-questions{args.repo_suffix}"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(OUTPUT_PATH),
        path_in_repo="wildchat_oracle_questions.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
