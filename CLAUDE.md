# GRPO Activation Oracle Training

## Project Overview
GRPO (Group Relative Policy Optimization) training for Activation Oracles - models that read internal activations of LLMs and answer questions with calibrated confidence scores.

## Key Files
- `grpo_ao/grpo_trainer.py` - Main GRPO trainer with steering hooks
- `grpo_ao/config.py` - Config + judge prompt
- `grpo_ao/judge.py` - LLM judge for informativeness scoring
- `grpo_ao/reward.py` - Brier score-based reward computation
- `grpo_ao/epistemic_status.py` - Output format parsing (`[epistemic status: X]` where X is 0-10)
- `train.py` - Training entry point
- `sft_format.py` - SFT data generation for format learning
- `observe_rollout.py` - Vibes check script (run rollouts without training)

## Output Format
```
[epistemic status: X] Answer
```
Where X is 0-10 (not 0-100). The confidence is normalized to 0-1 for Brier score computation.

## Current Config
- Model: Qwen/Qwen3-8B + LoRA
- Oracle LoRA (SFT-finetuned): `ceselder/qwen3-8b-oracle-sft-format`
  - This is the base oracle further trained with SFT to learn the `[epistemic status: X]` format
  - Requires system prompt (ORACLE_SYSTEM_PROMPT) to produce the format
  - `ceselder/activation-oracle-sft-epistemic` does NOT work (undertrained, 200 steps)
  - Without SFT checkpoint, the oracle outputs raw answers and gets 100% parse failures
- Generations: 8 per example
- Judge: GLM-4.7-flash via OpenRouter
- Reward: `informativeness - λ * (confidence/10 - informativeness)²`
- Dr. GRPO: `scale_rewards="none"`, `fix_length_bias=True`
- torch.compile: Enabled by default (use `--no_compile` if hooks break)

## Training on Remote GPU

### Quick setup (one command)
```bash
bash setup_remote.sh ssh1.vast.ai PORT
```
This syncs code, installs deps, verifies imports, and checks the GPU.

### Manual steps (if needed)
```bash
# 1. Sync code (nl_probes is vendored - no extra repos needed)
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'wandb' --exclude '*.pyc' --exclude '.mypy_cache' \
  -e "ssh -o StrictHostKeyChecking=no -p PORT" . root@HOST:/root/GRPO-activation-oracles/

# 2. Install deps (just pip packages, no editable installs needed)
ssh -o StrictHostKeyChecking=no -p PORT root@HOST \
  "pip install -q torch transformers peft datasets openai httpx nest-asyncio wandb tqdm huggingface_hub pydantic"

# 3. Vibes check (run in background, model download takes ~5min first time)
ssh -o StrictHostKeyChecking=no -p PORT root@HOST \
  "cd /root/GRPO-activation-oracles && nohup python3 observe_rollout.py --num_examples 2 --num_generations 4 > vibes.log 2>&1 &"
# Monitor: ssh ... "tail -f /root/vibes.log"

# 4. Start training
ssh -o StrictHostKeyChecking=no -p PORT root@HOST \
  "cd /root/GRPO-activation-oracles && nohup python3 train.py --no_eval > train.log 2>&1 &"
# Monitor: ssh ... "tail -f /root/GRPO-activation-oracles/train.log"
```

### Important notes for Claude Code
- **nl_probes is vendored** in `nl_probes/` - no need to clone/install the activation_oracles repo
- **Never** run `pip install -e .` on the activation_oracles repo - it pulls in vllm/flash-attn/ray and will crash the server
- Use `--no_eval` for training unless the full nl_probes package is installed (classification datasets need it)
- First `import torch` on a fresh server takes ~30-60s, be patient with SSH commands
- Use `-o StrictHostKeyChecking=no` for SSH to avoid host key issues on fresh VMs
- Run long commands via `nohup ... &` and monitor with `tail -f` to avoid SSH timeout kills
- The `.env` file must exist with HF_TOKEN, WANDB_API_KEY, OPENROUTER_API_KEY

## Inference Speed Notes
- HuggingFace `generate()` is slow (~3-10x slower than vLLM)
- torch.compile can help (~2-4x speedup) but may break with steering hooks
- vLLM doesn't support arbitrary forward hooks (needed for activation steering)
- EasySteer (https://github.com/ZJU-REAL/EasySteer) integrates steering with vLLM but may need custom algorithm for additive steering
- Qwen3 reasoning is disabled via `enable_thinking=False`

## Oracle Input Format

**CRITICAL**: The oracle was trained with a specific input format. Must match exactly:

```python
# Prefix format (from nl_probes.utils.dataset_utils)
SPECIAL_TOKEN = " ?"  # Space + question mark
prefix = f"Layer: {layer}\n{SPECIAL_TOKEN * num_positions} \n"

# Message format - NO SYSTEM PROMPT
messages = [
    {"role": "user", "content": prefix + question},
    {"role": "assistant", "content": answer}
]
```

Key points:
- **NO system prompt** - the oracle wasn't trained with one
- **Extraction layer**: 25/50/75% of model depth (e.g., layers 9, 18, 27 for 36-layer Qwen3-8B)
- **Injection layer**: Always layer 1 (oracle injection layer)
- **num_positions**: 1-20 (number of ` ?` tokens for activation injection)
- **Steering**: Activations injected at ` ?` token positions via forward hooks
- **Norm matching formula**: `h' = h + ||h|| * (v / ||v||)` - normalized steering scaled by original activation norm, then added

Reference: https://github.com/ceselder/dreaming-vectors/blob/main/find_redteam_vectors.py

## SFT Format Training
Before GRPO, the oracle needs to learn the `[epistemic status: X]` format via SFT.

### How it works (sft_format.py)
1. **Data generation**: For each example:
   - Extract activations from layer 25/50/75% of subject model (adapter disabled)
   - Inject at layer 1 of oracle (adapter enabled) using norm-matched steering
   - Generate oracle's natural answer
   - Prepend `[epistemic status: X]` with uniform random X (0-10)
2. **Training**: For each step:
   - Re-extract activations from subject model
   - Inject at layer 1 during forward pass
   - Train on formatted output with ORACLE_SYSTEM_PROMPT

### System Prompt (required at inference!)
```
Respond with: [epistemic status: X] Answer

X is your confidence from 0-10. Use the FULL range:
- 10: absolutely certain
- 8: very confident
...
```
See `grpo_ao/epistemic_status.py` for full prompt.

### TODO
- [ ] Re-run SFT with 2+ epochs (1000+ steps for 500 examples)
- [ ] Current run used 200 steps = 0.4 epochs (too few!)
- [ ] Push working checkpoint to HuggingFace
- [ ] Create requirements.txt

Scripts:
- `sft_format.py` - SFT with real steering injections (use this!)
- `sft_format_simple.py` - SFT without injections (deprecated)

## Key Papers
- Activation Oracles: https://arxiv.org/abs/2512.15674
- Dr. GRPO: https://arxiv.org/abs/2503.20783 (fixes length bias + std normalization)

## WandB
Project: https://wandb.ai/celestedeschamphelaere-personal/grpo-activation-oracle
