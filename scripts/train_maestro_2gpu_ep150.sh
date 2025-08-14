#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py \
  distributed.world_size=2 \
  common.max_epoch=150 \
  checkpoint.save_folder=/root/tf-logs/encodec_maestro_2gpus_ep150_4 \
  hydra.run.dir=/root/autodl-tmp/output/encodec_maestro_2gpus_ep150_4


