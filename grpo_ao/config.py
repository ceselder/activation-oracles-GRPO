"""Configuration for GRPO Activation Oracle training."""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """GRPO training configuration."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B"
    hook_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])

    # LoRA (if no pretrained checkpoint)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # GRPO
    num_train_steps: int = 1000
    num_generations: int = 8  # G responses per (prompt, question)
    kl_penalty: float = 0.05
    calibration_lambda: float = 1.0  # Increased from 0.5 to penalize hedging at 50
    oracle_temperature: float = 0.9

    # Training
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    max_new_tokens: int = 80

    # Logging
    log_samples_every: int = 10
    checkpoint_every: int = 100
    wandb_project: str = "grpo-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"
    push_to_hub: bool = True
    hub_repo_id: str = "ceselder/grpo-activation-oracle-qwen3-8b"

    # Judge (Gemini 3 Flash via OpenRouter with CoT)
    judge_model: str = "google/gemini-3-flash-preview"
    judge_max_tokens: int = 1000  # Allow CoT reasoning
    judge_thinking_level: str = "low"  # minimal/low/medium/high
    judge_temperature: float = 0.0

    # Evaluation
    eval_at_checkpoints: bool = True
    eval_datasets: list[str] | None = None  # None = all 9 datasets
    eval_batch_size: int = 4

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"grpo_{model_short}_g{self.num_generations}"


JUDGE_PROMPT = """Rate how informative and correct this answer is about the text.

TEXT: {prompt}

QUESTION: {question}

ANSWER: {answer}

Score from -100 to 100:
- -100 to -50: Confidently wrong, harmful misinformation
- -50 to 0: Wrong or nonsensical
- 0 to 20: Vague hedge that could apply to anything
- 20 to 40: Partially correct but missing key details
- 40 to 60: Correct with some relevant detail
- 60 to 80: Correct with good specificity
- 80 to 100: Excellent - correct, specific, captures nuance

Think through your reasoning, then output your final score as an integer on the last line."""
