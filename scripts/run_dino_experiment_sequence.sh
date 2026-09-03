#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/activate_dino.sh
export OMP_NUM_THREADS=8

REPO_ROOT=/root/PPM_CLIP_DINO_baseline
OUTPUT_BASE=/root/autodl-tmp/outputs/dinov2_vitl14_baseline
TEST_ROOT=/root/autodl-tmp/datasets/GenImage/genimage_test/test

TEST_SETS=(
  adm_imagenet
  biggan_imagenet
  glide_imagenet
  midjourney_imagenet
  sdv4_imagenet
  sdv5_imagenet
  vqdm_imagenet
  wukong_imagenet
)

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_BASE"

latest_checkpoint() {
  local checkpoint_dir=$1
  find "$checkpoint_dir" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

run_experiment() {
  local experiment_name=$1
  shift
  local output_root="$OUTPUT_BASE/$experiment_name"
  local train_marker="$output_root/TRAINING_FINISHED"
  local eval_marker="$output_root/GENIMAGE_EVAL_FINISHED"

  mkdir -p "$output_root"
  printf '%s\tSTART\t%s\n' "$(date --iso-8601=seconds)" "$experiment_name" \
    >> "$OUTPUT_BASE/sequence_status.tsv"

  if [[ ! -f "$train_marker" ]]; then
    python main.py \
      --gpu 0 \
      --batch_size 48 \
      --accumulation_steps 1 \
      --epochs 10 \
      --lr 1e-5 \
      --head_lr 1e-4 \
      --output_root "$output_root" \
      --test_sets stable_diffusion_v_1_4 \
      "$@"
    touch "$train_marker"
  fi

  local checkpoint
  checkpoint=$(latest_checkpoint "$output_root/checkpoints")
  if [[ -z "$checkpoint" ]]; then
    printf '%s\tNO_CHECKPOINT\t%s\n' "$(date --iso-8601=seconds)" "$experiment_name" \
      >> "$OUTPUT_BASE/sequence_status.tsv"
    return 0
  fi

  if [[ ! -f "$eval_marker" ]]; then
    python test.py \
      --gpu 0 \
      --batch_size 48 \
      --ckpt_path "$checkpoint" \
      --output_root "$output_root" \
      --test_root "$TEST_ROOT" \
      --test_sets "${TEST_SETS[@]}" \
      "$@"
    touch "$eval_marker"
  fi

  printf '%s\tCOMPLETE\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$experiment_name" "$checkpoint" \
    >> "$OUTPUT_BASE/sequence_status.tsv"
}

run_experiment linear_probe --freeze_backbone
run_experiment full_finetune

printf '%s\tSEQUENCE_COMPLETE\n' "$(date --iso-8601=seconds)" \
  >> "$OUTPUT_BASE/sequence_status.tsv"
