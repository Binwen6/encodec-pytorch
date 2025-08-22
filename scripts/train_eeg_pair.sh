#!/usr/bin/env bash
set -euo pipefail

# Single-GPU EEG→Audio training entrypoint.
# Edit paths or override via CLI as needed.

PY=/root/code/encodec-pytorch/train_multi_gpu.py

python "$PY" \
  distributed.data_parallel=False \
  datasets.type=eeg_pair \
  datasets.eeg_dir=/root/autodl-tmp/eeg_npy \
  datasets.audio_path=/root/autodl-tmp/music_dataset/11_24k.wav \
  datasets.tensor_cut=192000 datasets.train_ratio=0.8 datasets.batch_size=4 \
  model.sample_rate=24000 model.channels=1 \
  checkpoint.checkpoint_path=/root/autodl-tmp/checkpoint/encodec_maestro_4gpus_ep150_256_4/bs8_cut72000_length0_epoch150_lr0.0003.pt \
  optimization.lr=1e-4 optimization.disc_lr=1e-4 \
  common.max_epoch=20 \
  checkpoint.save_folder=/root/code/encodec-pytorch/checkpoints \
  tensorboard.log_dir=/root/code/encodec-pytorch/runs


