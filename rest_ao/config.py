"""Configuration for ReST Activation Oracle training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RESTConfig:
    """Configuration for Reinforced Self-Training of calibrated Activation Oracles."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B"
    hook_layer: int = 1  # Layer to inject activations (paper uses layer 1)
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])  # Extract at multiple depths

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Data settings
    num_prompts: int = 10_000  # Number of prompts to use
    questions_per_prompt: int = 10  # Questions generated per prompt
    question_temperature: float = 1.75  # Very high temp for maximum question diversity
    question_batch_size: int = 32  # Batch size for question generation (variable prompt lengths)
    grow_batch_size: int = 32  # Batch size for GROW phase
    judge_batch_size: int = 32  # Batch size for SCORE phase

    # ReST settings
    num_rest_rounds: int = 10  # Total rounds to run
    checkpoint_every: int = 2  # Save checkpoint every N rounds
    samples_per_question: int = 3  # Oracle responses sampled per question
    oracle_temperature: float = 1.2  # Higher for diverse samples (was 0.7)
    filter_bottom_percent: float = 0.2  # Remove bottom 20% by reward
    calibration_lambda: float = 0.5  # λ in reward formula

    # Training settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    epochs_per_round: int = 1

    # Generation settings
    max_new_tokens: int = 80  # Reduced from 150 for speed (epistemic status + short answer)

    # Logging
    wandb_project: str = "rest-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"
    cache_dir: str = "cache"

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Judge settings (OpenRouter for Qwen-32B or similar)
    judge_model: str = "qwen/qwen-2.5-72b-instruct"  # Via OpenRouter
    judge_temperature: float = 0.0

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"rest_{model_short}_r{self.num_rest_rounds}"


# Question generation templates for Activation Oracle training
# AO answers questions using only model activations (hidden states), not raw text
# Questions should probe what the model internally represents about the text
_AO_CONTEXT = """You are generating questions to train an Activation Oracle - a model that answers questions about text using only internal activations (hidden states), not the raw text.

Good questions probe what the model internally represents:
- Topic/theme: "What is this text about?", "Is this related to technology?"
- User traits: "Does the user seem technical?", "What's the user's likely expertise level?"
- Sentiment/tone: "Is the tone positive or negative?", "Does the user seem frustrated?"
- Intent: "Is this a request for help?", "Is this creative writing or factual?"
- Content: "Is a person mentioned?", "What language is this written in?"
- Meta: "Is this a short or long message?", "Is there code in this text?"

Bad questions:
- Require external knowledge to verify correctness
- Are too specific (exact quotes, word counts)
- Are vague or unanswerable

Mix yes/no questions with open-ended ones. All questions MUST be in English.
"""

QUESTION_TEMPLATES = [
    _AO_CONTEXT + """Generate questions about this text:
- 2-3 yes/no questions about topic, sentiment, or user traits
- 2-3 open-ended questions about theme, intent, or content
Examples: "Is this about science?", "What is the user asking for?", "Does the user seem upset?"

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create questions probing what the model represents:
- Topic detection: "What subject area is this?", "Is this technical content?"
- User inference: "Is this likely written by an expert?", "What might the user's goal be?"
- Sentiment: "What's the emotional tone?", "Is this positive or negative?"

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write questions about internal representations:
- Yes/no: "Is this a question?", "Is code mentioned?", "Is the tone formal?"
- Open: "What topic is this about?", "What does the user want?", "Summarize the intent."

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate questions testing latent knowledge:
- Theme/topic questions (open-ended)
- Sentiment/tone questions (yes/no or short answer)
- User characteristic inference (expertise, mood, intent)

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create inference questions:
- "What is the main topic?" (open)
- "Is this a technical question?" (yes/no)
- "What does the user want to accomplish?" (open)
- "Is the user frustrated or satisfied?" (yes/no or short)

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write questions about themes and intent:
- Topic: "What field/domain is this about?"
- Intent: "Is this asking for help, information, or something else?"
- Tone: "How would you describe the sentiment?"
Mix yes/no with open-ended.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate questions a model's activations could answer:
- "What language is this?"
- "Is this formal or casual writing?"
- "What is the user trying to do?"
- "Is there a question being asked?"
- "What's the general subject matter?"

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create questions about text characteristics:
- Content type: "Is this code?", "Is this a conversation?"
- Topic: "What is this about?"
- User: "What can you infer about who wrote this?"
- Sentiment: "What's the emotional tone?"

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write probing questions:
- Yes/no: topic checks, sentiment checks, format checks
- Open: theme description, intent summary, user inference
Avoid questions needing external verification.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate varied questions:
- "Is this text about [topic]?" (yes/no)
- "What is the main theme?" (open)
- "Does the user seem [trait]?" (yes/no)
- "What is the user's intent?" (open)
- "Is the tone [sentiment]?" (yes/no)

TEXT:
{prompt}""",
]


JUDGE_PROMPT = """Rate how informative and correct this answer is on a scale from 0.0 to 1.0.

ORIGINAL TEXT:
{prompt}

QUESTION: {question}

ANSWER: {answer}

Scoring guidelines:
- 0.0 = Completely wrong or nonsensical
- 0.2 = Technically not wrong but extremely vague (e.g., "it's about science")
- 0.4 = Partially correct with minimal detail
- 0.6 = Correct with reasonable detail
- 0.8 = Correct with good specificity and relevant details
- 1.0 = Correct, highly specific, captures nuance and key information

Consider both correctness AND level of detail. A vague answer that is technically true should score low.

Return only a single decimal number between 0.0 and 1.0."""
