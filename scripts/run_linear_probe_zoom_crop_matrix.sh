#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/activate_dino.sh
export OMP_NUM_THREADS=8

REPO_ROOT=/root/PPM_CLIP_DINO_baseline
OUTPUT_BASE=/root/autodl-tmp/outputs/dinov2_vitl14_baseline/linear_probe_zoom_crop_matrix
GENIMAGE_TEST_ROOT=/root/autodl-tmp/datasets/GenImage/genimage_test/test
CHAMELEON_TEST_ROOT=/root/autodl-tmp/datasets/Chameleon/Chameleon/test
STATUS_FILE="$OUTPUT_BASE/matrix_status.tsv"

HEAD_LRS=(3e-5 3e-4)
VAL_SCOPES=(train_val train_only)
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
  local val_scope=$2
  local lr_tag=${head_lr//-/m}
  lr_tag=${lr_tag//+/p}
  lr_tag=${lr_tag//./d}

  local output_root="$OUTPUT_BASE/headlr_${lr_tag}_${val_scope}"
  local genimage_output="$output_root/genimage_eval"
  local chameleon_output="$output_root/chameleon_eval"
  local val_flag=()
  if [[ "$val_scope" == train_val ]]; then
    val_flag=(--zoom_crop_on_val)
  fi
  current_run="head_lr=$head_lr zoom_scope=$val_scope"

  mkdir -p "$output_root"
  printf '%s\n' \
    "mode=linear_probe" \
    "backbone=dinov2_vitl14" \
    "freeze_backbone=true" \
    "batch_size=48" \
    "accumulation_steps=2" \
    "effective_batch_size=96" \
    "dropout=0.0" \
    "head_lr=$head_lr" \
    "backbone_lr=1e-5 (inactive because backbone is frozen)" \
    "zoom_crop_prob=0.5" \
    "zoom_crop_scale=2.0" \
    "zoom_crop_scope=$val_scope" \
    "epochs=10 (maximum; early stopping may stop sooner)" \
    "seed=1029" \
    > "$output_root/EXPERIMENT_CONFIG.txt"

  if [[ ! -f "$output_root/TRAINING_FINISHED" ]]; then
    printf '%s\tTRAIN_START\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
    python main.py \
      --gpu 0 \
      --epochs 10 \
      --head_lr "$head_lr" \
      --dropout 0.0 \
      --zoom_crop_prob 0.5 \
      --zoom_crop_scale 2.0 \
      "${val_flag[@]}" \
      --output_root "$output_root" \
      --test_sets stable_diffusion_v_1_4 \
      --freeze_backbone \
      2>&1 | tee -a "$output_root/train_console.log"
    touch "$output_root/TRAINING_FINISHED"
    printf '%s\tTRAIN_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
  fi

  local checkpoint
  checkpoint=$(latest_checkpoint "$output_root/checkpoints")
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    printf '%s\tNO_CHECKPOINT\t%s\n' "$(date --iso-8601=seconds)" "$current_run" >> "$STATUS_FILE"
    return 1
  fi

  if [[ ! -f "$output_root/GENIMAGE_EVAL_FINISHED" ]]; then
    printf '%s\tGENIMAGE_EVAL_START\t%s\t%s\n' "$(date --iso-8601=seconds)" "$current_run" "$checkpoint" >> "$STATUS_FILE"
    python test.py \
      --gpu 0 \
      --head_lr "$head_lr" \
      --dropout 0.0 \
      --zoom_crop_prob 0.5 \
      --zoom_crop_scale 2.0 \
      "${val_flag[@]}" \
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
      --head_lr "$head_lr" \
      --dropout 0.0 \
      --zoom_crop_prob 0.5 \
      --zoom_crop_scale 2.0 \
      "${val_flag[@]}" \
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

printf '%s\tMATRIX_START\thead_lrs=%s\tval_scopes=%s\n' \
  "$(date --iso-8601=seconds)" "${HEAD_LRS[*]}" "${VAL_SCOPES[*]}" >> "$STATUS_FILE"
for head_lr in "${HEAD_LRS[@]}"; do
  for val_scope in "${VAL_SCOPES[@]}"; do
    run_one "$head_lr" "$val_scope"
  done
done
printf '%s\tMATRIX_COMPLETE\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
touch "$OUTPUT_BASE/MATRIX_FINISHED"
