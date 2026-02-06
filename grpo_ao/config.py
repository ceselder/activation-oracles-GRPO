"""Configuration for GRPO Activation Oracle training."""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """GRPO training configuration."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "ceselder/qwen3-8b-oracle-sft-format"
    hook_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [50])  # Only middle layer

    # LoRA (if no pretrained checkpoint)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # GRPO
    num_train_steps: int = 1000
    num_generations: int = 8  # Reduced - no gradient checkpointing due to hook conflict
    examples_per_batch: int = 4  # 8×4=32 rollouts per batch (fits in 80GB A100)
    kl_penalty: float = 0.04
    calibration_lambda: float = 1.0  # Brier score weight (was 0.75, increased for calibration pressure)
    confidence_entropy_bonus: float = 0.15  # Bonus for confidence diversity within group
    oracle_temperature: float = 1.2  # Slightly higher for more exploration
    # Dr. GRPO: "none" = don't scale by std (recommended), "group" = original GRPO
    scale_rewards: str = "none"
    # Dr. GRPO length bias fix: normalize by global constant (max_tokens * G) not response length
    fix_length_bias: bool = True

    # Training
    learning_rate: float = 3e-6
    max_grad_norm: float = 1.0
    max_new_tokens: int = 300  # Allow longer AO responses
    gradient_accumulation_steps: int = 2  # 4 examples × 2 accum = 8 effective batch

    # Logging
    log_samples_every: int = 10
    checkpoint_every: int = 100
    wandb_project: str = "grpo-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"
    push_to_hub: bool = True
    hub_repo_id: str = "ceselder/grpo-activation-oracle-qwen3-8b"

    # Judge (Gemini 2.5 Flash Lite via OpenRouter - ultra fast + cheap)
    judge_model: str = "google/gemini-2.5-flash-lite"
    judge_max_tokens: int = 20  # Just the score number, no CoT
    judge_thinking_level: str = "low"  # minimal/low/medium/high
    judge_temperature: float = 0.5

    # Evaluation
    eval_at_checkpoints: bool = True
    eval_datasets: list[str] | None = None  # None = all 9 datasets
    eval_batch_size: int = 4

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_torch_compile: bool = False  # torch.compile causes recompilation hell with variable-length inputs

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"grpo_{model_short}_g{self.num_generations}"


JUDGE_PROMPT = """An Activation Oracle (AO) is a novel method that reads the internal neural activations (hidden states) of a language model and describes what the model is "thinking" in natural language. It cannot see the text directly - only the model's internal representations.

The AO read activations from a model processing: {prompt}
Question about the model's internal state: {question}
AO's answer: {answer}

Rate correctness 0-100. 0-20=wrong, 20-40=generic, 40-60=correct but terse, 60-80=correct+detailed, 80-100=insightful. Correct "Yes"/"No" scores 40-50; explaining WHY scores higher.

Score:"""
