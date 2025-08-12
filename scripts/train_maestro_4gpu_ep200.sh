#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi_gpu.py \
  distributed.world_size=4 \
  common.max_epoch=200 \
  checkpoint.save_folder=/root/autodl-tmp/checkpoint/encodec_maestro_4gpus_ep200 \
  hydra.run.dir=/root/autodl-tmp/output/encodec_maestro_4gpus_ep200


