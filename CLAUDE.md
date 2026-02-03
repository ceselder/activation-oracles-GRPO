# GRPO Activation Oracle Training

## Project Overview
GRPO (Group Relative Policy Optimization) training for Activation Oracles - models that read internal activations of LLMs and answer questions with calibrated confidence scores.

## Key Files
- `grpo_ao/grpo_trainer.py` - Main GRPO trainer
- `grpo_ao/config.py` - Config + judge prompt
- `grpo_ao/judge.py` - LLM judge for informativeness scoring
- `grpo_ao/reward.py` - Brier score-based reward computation
- `grpo_ao/epistemic_status.py` - Output format parsing
- `train.py` - Entry point
- `sft_format.py` - SFT data generation

## Training on Remote GPU
```bash
# SSH to vast.ai server
ssh -p PORT root@IP

# Sync code
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'wandb' -e "ssh -p PORT" . root@IP:/root/GRPO-activation-oracles/

# Start training
nohup python3 train.py > train.log 2>&1 &

# Monitor
tail -f train.log
```

## Current Config
- Model: Gemma 2 9B + LoRA
- Generations: 8 per example
- Judge: GLM-4.7-flash via OpenRouter
- Reward: `informativeness - λ * (confidence/100 - informativeness)²`

## WandB
Project: https://wandb.ai/celestedeschamphelaere-personal/grpo-activation-oracle
