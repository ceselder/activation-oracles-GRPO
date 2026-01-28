"""Data pipeline for ReST training.

Handles:
1. Loading diverse text prompts (RedPajama/The Pile subsets)
2. Generating questions with diverse templates
3. Extracting activations
4. Creating training samples
"""

import random
from dataclasses import dataclass, field
from typing import Iterator

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from rest_ao.question_generation import QuestionGenerator, GeneratedQuestions


@dataclass
class PromptQuestionPair:
    """A prompt with its associated question and activation info."""

    prompt: str
    question: str
    layer: int
    context_input_ids: list[int]
    context_positions: list[int]
    template_idx: int


@dataclass
class OracleSample:
    """A sample for oracle training/generation."""

    prompt: str
    question: str
    oracle_response: str  # Full response with epistemic status
    informativeness: float | None = None
    reward: float | None = None


def load_diverse_prompts(
    num_prompts: int = 10000,
    max_length: int = 1024,
    seed: int = 42,
) -> list[str]:
    """Load diverse text prompts from multiple sources.

    Uses a mix of:
    - Wikipedia (factual)
    - News articles (current events)
    - ArXiv abstracts (technical)
    - Reddit (conversational)
    - Fiction (creative)

    Args:
        num_prompts: Total number of prompts to load
        max_length: Maximum character length per prompt
        seed: Random seed

    Returns:
        List of diverse text prompts
    """
    random.seed(seed)
    prompts = []

    # Distribution of sources
    sources = [
        ("wikipedia", 0.25),
        ("c4", 0.25),
        ("arxiv", 0.15),
        ("github", 0.15),
        ("stackexchange", 0.1),
        ("books", 0.1),
    ]

    for source, fraction in sources:
        n = int(num_prompts * fraction)
        source_prompts = _load_source(source, n, max_length, seed)
        prompts.extend(source_prompts)

    # Shuffle and trim to exact count
    random.shuffle(prompts)
    prompts = prompts[:num_prompts]

    return prompts


def _load_source(source: str, n: int, max_length: int, seed: int) -> list[str]:
    """Load prompts from a specific source."""
    prompts = []

    try:
        if source == "wikipedia":
            ds = load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i >= n * 2:  # Load extra to filter
                    break
                text = item["text"][:max_length]
                if len(text) > 100:  # Skip very short
                    prompts.append(text)

        elif source == "c4":
            ds = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i >= n * 2:
                    break
                text = item["text"][:max_length]
                if len(text) > 100:
                    prompts.append(text)

        elif source == "arxiv":
            ds = load_dataset(
                "ccdv/arxiv-summarization",
                split="train",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i >= n * 2:
                    break
                text = item["article"][:max_length]
                if len(text) > 100:
                    prompts.append(text)

        elif source == "github":
            ds = load_dataset(
                "codeparrot/github-code",
                streaming=True,
                split="train",
                languages=["Python"],
            )
            for i, item in enumerate(ds):
                if i >= n * 2:
                    break
                text = item["code"][:max_length]
                if len(text) > 100:
                    prompts.append(text)

        elif source == "stackexchange":
            ds = load_dataset(
                "flax-sentence-embeddings/stackexchange_title_body_jsonl",
                split="train",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i >= n * 2:
                    break
                text = f"{item['title']}\n\n{item['body']}"[:max_length]
                if len(text) > 100:
                    prompts.append(text)

        elif source == "books":
            ds = load_dataset(
                "bookcorpus",
                split="train",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i >= n * 2:
                    break
                text = item["text"][:max_length]
                if len(text) > 100:
                    prompts.append(text)

    except Exception as e:
        print(f"Warning: Could not load {source}: {e}")
        # Fall back to C4 if source fails
        if source != "c4":
            return _load_source("c4", n, max_length, seed)

    random.shuffle(prompts)
    return prompts[:n]


def create_prompt_question_pairs(
    prompts: list[str],
    question_generator: QuestionGenerator,
    tokenizer: PreTrainedTokenizer,
    layer_percents: list[int],
    model_name: str,
    questions_per_prompt: int = 10,
) -> Iterator[PromptQuestionPair]:
    """Create prompt-question pairs with activation positions.

    Args:
        prompts: List of text prompts
        question_generator: Generator for questions
        tokenizer: Tokenizer for the model
        layer_percents: Layers to sample from (e.g., [25, 50, 75])
        model_name: Model name for layer calculation
        questions_per_prompt: Target questions per prompt

    Yields:
        PromptQuestionPair objects
    """
    from nl_probes.utils.common import layer_percent_to_layer

    layers = [layer_percent_to_layer(model_name, p) for p in layer_percents]

    for prompt in tqdm(prompts, desc="Generating questions"):
        # Generate questions
        gen_result = question_generator.generate_questions(prompt)
        questions = gen_result.questions[:questions_per_prompt]

        if not questions:
            continue

        # Tokenize the prompt for activation extraction
        context_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Use all positions after some initial context
        min_offset = max(1, len(context_ids) // 4)
        context_positions = list(range(min_offset, len(context_ids)))

        if not context_positions:
            context_positions = list(range(len(context_ids)))

        for question in questions:
            layer = random.choice(layers)

            yield PromptQuestionPair(
                prompt=prompt,
                question=question,
                layer=layer,
                context_input_ids=context_ids,
                context_positions=context_positions,
                template_idx=gen_result.template_idx,
            )
