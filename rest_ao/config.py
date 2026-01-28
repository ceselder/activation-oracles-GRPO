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
    num_prompts: int = 100  # Fewer prompts for faster feedback loops
    questions_per_prompt: int = 20  # More questions per prompt
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
- Super open-ended: "What is the model thinking about?", "What are the key themes here?"
- Topic/theme: "What is this text about?", "Is this related to technology?"
- User traits: "Does the user seem technical?", "What's the user's likely expertise level?"
- Sentiment/tone: "Is the tone positive or negative?", "Does the user seem frustrated?"
- Intent: "Is this a request for help?", "Is this creative writing or factual?"
- Content: "Is a person mentioned?", "What can you infer about the context?"
- Meta: "Is this a short or long message?", "Is there code in this text?"

Bad questions:
- Require external knowledge to verify correctness
- Are too specific (exact quotes, word counts)

IMPORTANT: Always include at least one super open-ended question like "What is the model thinking about?" or "Describe what this text is about in detail."
Mix yes/no questions with open-ended ones. All questions MUST be in English.
"""

QUESTION_TEMPLATES = [
    _AO_CONTEXT + """Generate 15-20 questions about this text.

MUST INCLUDE:
- 1-2 super open-ended: "What is the model thinking about?", "Describe the overall content and themes."
- 3-4 topic/theme questions (mix of yes/no and open)
- 3-4 user inference questions (expertise, intent, mood)
- 3-4 sentiment/tone questions
- 2-3 meta questions (format, style, type of content)

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate 15-20 diverse questions probing latent representations.

Include:
- "What is the model thinking about?" (always include this one)
- "What are the main themes and ideas here?"
- Topic detection questions
- User trait inference questions
- Sentiment analysis questions
- Content type questions

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create 15-20 questions for activation oracle training.

Required question types:
1. Super open-ended (2-3): "What is this text about in detail?", "What is the model's overall impression?"
2. Yes/no topic checks (3-4): "Is this about X?", "Does this mention Y?"
3. User inference (3-4): "What can you infer about the author?", "What's the user's goal?"
4. Sentiment (3-4): "Is the tone positive?", "Does the user seem frustrated?"
5. Meta (2-3): "Is this formal writing?", "Is there code?"

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write 15-20 questions testing what the model internally represents.

Always start with: "What is the model thinking about this text?"

Then add:
- Open-ended theme questions
- Yes/no topic verification
- User characteristic inference
- Emotional tone analysis
- Content format questions

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate 15-20 activation probing questions.

Must include these types:
- "What is the model thinking about?" (required)
- "Summarize what this text is about" (required)
- Topic questions (is this about X?)
- Intent questions (what does the user want?)
- Sentiment questions (positive/negative/neutral?)
- Author inference (expert? frustrated? curious?)

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
