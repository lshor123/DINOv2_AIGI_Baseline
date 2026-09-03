#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/activate_dino.sh
cd /root/PPM_CLIP_DINO_baseline

python main.py \
  --gpu 0 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --epochs 10 \
  --lr 1e-5 \
  --head_lr 1e-4 \
  --test_sets stable_diffusion_v_1_4
