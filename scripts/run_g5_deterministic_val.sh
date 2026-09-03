#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/activate_dino.sh
export OMP_NUM_THREADS=8

REPO_ROOT=/root/PPM_CLIP_DINO_baseline
OUTPUT_ROOT=/root/autodl-tmp/outputs/dinov2_vitl14_baseline/linear_probe_zoom_crop_matrix/G5_headlr_3em4_train_only_deterministic_val
GENIMAGE_TEST_ROOT=/root/autodl-tmp/datasets/GenImage/genimage_test/test
CHAMELEON_TEST_ROOT=/root/autodl-tmp/datasets/Chameleon/Chameleon/test
STATUS_FILE="$OUTPUT_ROOT/g5_status.tsv"
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

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
stage=initialization
trap 'printf "%s\tFAILED\t%s\tline=%s\n" "$(date --iso-8601=seconds)" "$stage" "$LINENO" >> "$STATUS_FILE"' ERR

latest_checkpoint() {
  find "$OUTPUT_ROOT/checkpoints" -maxdepth 1 -type f -name '*.pt' -printf '%T@ %p\n' \
    | sort -n | tail -n 1 | cut -d' ' -f2-
}

printf '%s\n' \
  'experiment=G5' \
  'reference=G4 with validation transform changed only' \
  'mode=linear_probe' \
  'backbone=dinov2_vitl14' \
  'freeze_backbone=true' \
  'batch_size=48' \
  'accumulation_steps=2' \
  'effective_batch_size=96' \
  'dropout=0.0' \
  'head_lr=3e-4' \
  'backbone_lr=1e-5 (inactive because backbone is frozen)' \
  'train_transform=RandomZoomCrop(probability=0.5, scale=2.0) + RandomHorizontalFlip' \
  'validation_transform=test_transform (size check + CenterCrop(224), no random augmentation)' \
  'epochs=10 (maximum; original early stopping may stop sooner)' \
  'seed=1029' \
  > "$OUTPUT_ROOT/EXPERIMENT_CONFIG.txt"

stage=training
printf '%s\tTRAIN_START\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
python main.py \
  --gpu 0 \
  --epochs 10 \
  --head_lr 3e-4 \
  --dropout 0.0 \
  --zoom_crop_prob 0.5 \
  --zoom_crop_scale 2.0 \
  --deterministic_val \
  --output_root "$OUTPUT_ROOT" \
  --test_sets stable_diffusion_v_1_4 \
  --freeze_backbone \
  2>&1 | tee -a "$OUTPUT_ROOT/train_console.log"
touch "$OUTPUT_ROOT/TRAINING_FINISHED"
printf '%s\tTRAIN_COMPLETE\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"

checkpoint=$(latest_checkpoint)
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  printf '%s\tNO_CHECKPOINT\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
  exit 1
fi
printf '%s\tCHECKPOINT\tG5\t%s\n' "$(date --iso-8601=seconds)" "$checkpoint" >> "$STATUS_FILE"

stage=genimage_eval
printf '%s\tGENIMAGE_EVAL_START\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
python test.py \
  --gpu 0 \
  --head_lr 3e-4 \
  --dropout 0.0 \
  --zoom_crop_prob 0.5 \
  --zoom_crop_scale 2.0 \
  --deterministic_val \
  --ckpt_path "$checkpoint" \
  --output_root "$OUTPUT_ROOT/genimage_eval" \
  --test_root "$GENIMAGE_TEST_ROOT" \
  --test_sets "${TEST_SETS[@]}" \
  --freeze_backbone \
  2>&1 | tee -a "$OUTPUT_ROOT/genimage_eval_console.log"
touch "$OUTPUT_ROOT/GENIMAGE_EVAL_FINISHED"
printf '%s\tGENIMAGE_EVAL_COMPLETE\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"

stage=chameleon_eval
printf '%s\tCHAMELEON_EVAL_START\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
python test.py \
  --gpu 0 \
  --head_lr 3e-4 \
  --dropout 0.0 \
  --zoom_crop_prob 0.5 \
  --zoom_crop_scale 2.0 \
  --deterministic_val \
  --ckpt_path "$checkpoint" \
  --output_root "$OUTPUT_ROOT/chameleon_eval" \
  --test_root "$CHAMELEON_TEST_ROOT" \
  --test_sets . \
  --freeze_backbone \
  2>&1 | tee -a "$OUTPUT_ROOT/chameleon_eval_console.log"
touch "$OUTPUT_ROOT/CHAMELEON_EVAL_FINISHED"
printf '%s\tCHAMELEON_EVAL_COMPLETE\tG5\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"

stage=complete
touch "$OUTPUT_ROOT/G5_FINISHED"
printf '%s\tRUN_COMPLETE\tG5\t%s\n' "$(date --iso-8601=seconds)" "$checkpoint" >> "$STATUS_FILE"
