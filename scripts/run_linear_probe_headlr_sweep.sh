#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/activate_dino.sh
export OMP_NUM_THREADS=8

REPO_ROOT=/root/PPM_CLIP_DINO_baseline
OUTPUT_BASE=/root/autodl-tmp/outputs/dinov2_vitl14_baseline/linear_probe_accum2_headlr_sweep
GENIMAGE_TEST_ROOT=/root/autodl-tmp/datasets/GenImage/genimage_test/test
CHAMELEON_TEST_ROOT=/root/autodl-tmp/datasets/Chameleon/Chameleon/test
STATUS_FILE="$OUTPUT_BASE/sweep_status.tsv"

HEAD_LRS=(3e-5 1e-4 3e-4)
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

mkdir -p "$OUTPUT_BASE"
cd "$REPO_ROOT"

current_run=initialization
trap 'printf "%s\tFAILED\t%s\tline=%s\n" "$(date --iso-8601=seconds)" "$current_run" "$LINENO" >> "$STATUS_FILE"' ERR

latest_checkpoint() {
  local checkpoint_dir=$1
  find "$checkpoint_dir" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

run_one() {
  local head_lr=$1
  local tag=${head_lr//-/m}
  tag=${tag//+/p}
  tag=${tag//./d}
  local output_root="$OUTPUT_BASE/headlr_${tag}"
  local genimage_output="$output_root/genimage_eval"
  local chameleon_output="$output_root/chameleon_eval"
  current_run="head_lr=$head_lr"

  mkdir -p "$output_root"
  printf '%s\n' \
    "mode=linear_probe" \
    "batch_size=48" \
    "accumulation_steps=2" \
    "effective_batch_size=96" \
    "head_lr=$head_lr" \
    "backbone_lr=1e-5 (inactive because backbone is frozen)" \
    "epochs=5 (applies to remaining runs; completed earlier runs are not repeated)" \
    "seed=1029" \
    > "$output_root/EXPERIMENT_CONFIG.txt"

  if [[ ! -f "$output_root/TRAINING_FINISHED" ]]; then
    printf '%s\tTRAIN_START\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
    python main.py \
      --gpu 0 \
      --batch_size 48 \
      --accumulation_steps 2 \
      --epochs 5 \
      --lr 1e-5 \
      --head_lr "$head_lr" \
      --output_root "$output_root" \
      --test_sets stable_diffusion_v_1_4 \
      --freeze_backbone \
      2>&1 | tee -a "$output_root/train_console.log"
    touch "$output_root/TRAINING_FINISHED"
    printf '%s\tTRAIN_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
  fi

  local checkpoint
  checkpoint=$(latest_checkpoint "$output_root/checkpoints")
  if [[ -z "$checkpoint" ]]; then
    printf '%s\tNO_CHECKPOINT\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
    return 1
  fi

  if [[ ! -f "$output_root/GENIMAGE_EVAL_FINISHED" ]]; then
    printf '%s\tGENIMAGE_EVAL_START\t%s\t%s\n' "$(date --iso-8601=seconds)" "$current_run" "$checkpoint" >> "$STATUS_FILE"
    python test.py \
      --gpu 0 \
      --batch_size 48 \
      --accumulation_steps 2 \
      --ckpt_path "$checkpoint" \
      --output_root "$genimage_output" \
      --test_root "$GENIMAGE_TEST_ROOT" \
      --test_sets "${TEST_SETS[@]}" \
      --freeze_backbone \
      2>&1 | tee -a "$output_root/genimage_eval_console.log"
    touch "$output_root/GENIMAGE_EVAL_FINISHED"
    printf '%s\tGENIMAGE_EVAL_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
  fi

  if [[ ! -f "$output_root/CHAMELEON_EVAL_FINISHED" ]]; then
    printf '%s\tCHAMELEON_EVAL_START\t%s\t%s\n' "$(date --iso-8601=seconds)" "$current_run" "$checkpoint" >> "$STATUS_FILE"
    python test.py \
      --gpu 0 \
      --batch_size 48 \
      --accumulation_steps 2 \
      --ckpt_path "$checkpoint" \
      --output_root "$chameleon_output" \
      --test_root "$CHAMELEON_TEST_ROOT" \
      --test_sets . \
      --freeze_backbone \
      2>&1 | tee -a "$output_root/chameleon_eval_console.log"
    touch "$output_root/CHAMELEON_EVAL_FINISHED"
    printf '%s\tCHAMELEON_EVAL_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
  fi

  printf '%s\tRUN_COMPLETE\t%s\t%s\n' "$(date --iso-8601=seconds)" "$current_run" "$checkpoint" >> "$STATUS_FILE"
}

printf '%s\tSWEEP_START\thead_lrs=%s\n' "$(date --iso-8601=seconds)" "${HEAD_LRS[*]}" >> "$STATUS_FILE"
for head_lr in "${HEAD_LRS[@]}"; do
  run_one "$head_lr"
done
printf '%s\tSWEEP_COMPLETE\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
touch "$OUTPUT_BASE/SWEEP_FINISHED"
