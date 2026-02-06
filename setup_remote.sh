#!/bin/bash
# One-command remote GPU setup for GRPO Activation Oracle training.
# Usage: bash setup_remote.sh <ssh_host> <ssh_port>
# Example: bash setup_remote.sh ssh1.vast.ai 36489
set -e

HOST="${1:?Usage: bash setup_remote.sh <ssh_host> <ssh_port>}"
PORT="${2:?Usage: bash setup_remote.sh <ssh_host> <ssh_port>}"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="rsync -avz --exclude .git --exclude __pycache__ --exclude wandb --exclude '*.pyc' --exclude .mypy_cache -e 'ssh -o StrictHostKeyChecking=no -p $PORT'"
REMOTE_DIR="/root/GRPO-activation-oracles"

echo "=== Step 1: Sync code ==="
eval $SCP "$(dirname "$0")/" "root@$HOST:$REMOTE_DIR/"

echo "=== Step 2: Install dependencies ==="
$SSH "pip install -q torch transformers peft datasets openai httpx nest-asyncio wandb tqdm huggingface_hub pydantic 2>&1 | tail -5"

echo "=== Step 3: Verify imports ==="
$SSH "cd $REMOTE_DIR && python3 -c '
from nl_probes.utils.steering_hooks import add_hook
from nl_probes.utils.common import load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import get_introspection_prefix, SPECIAL_TOKEN
from grpo_ao.config import GRPOConfig
print(\"All imports OK\")
'"

echo "=== Step 4: Check GPU ==="
$SSH "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"

echo "=== Ready! ==="
echo "SSH in:  ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
echo "Vibes:   cd $REMOTE_DIR && python3 observe_rollout.py --num_examples 2 --num_generations 4"
echo "Train:   cd $REMOTE_DIR && nohup python3 train.py --no_eval > train.log 2>&1 &"
