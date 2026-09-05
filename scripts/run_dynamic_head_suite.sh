#!/usr/bin/env bash
set -Eeuo pipefail
source /root/autodl-tmp/activate_dino.sh
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONHASHSEED=1029
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${DYNAMIC_OUTPUT_ROOT:-/root/autodl-tmp/outputs/dinov2_vitl14_baseline/dynamic_head_20260905}
G5_CKPT=/root/autodl-tmp/outputs/dinov2_vitl14_baseline/linear_probe_zoom_crop_matrix/G5_headlr_3em4_train_only_deterministic_val/checkpoints/genimage_DINOv2_ViTL14_0902_1731.pt
GENIMAGE_ROOT=/root/autodl-tmp/datasets/GenImage/genimage_test/test
CHAMELEON_ROOT=/root/autodl-tmp/datasets/Chameleon/Chameleon/test
TEST_SETS=(adm_imagenet biggan_imagenet glide_imagenet midjourney_imagenet sdv4_imagenet sdv5_imagenet vqdm_imagenet wukong_imagenet)
mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
exec 9>"$OUTPUT_ROOT/suite.lock"
flock -n 9 || { echo 'Another suite owns this output directory'; exit 1; }
STATUS_FILE="$OUTPUT_ROOT/sequence_status.tsv"
stage=initialization
run=none
trap 'printf "%s\tFAILED\t%s\t%s\tline=%s\n" "$(date --iso-8601=seconds)" "$run" "$stage" "$LINENO" >> "$STATUS_FILE"' ERR
[[ -f "$G5_CKPT" ]]
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit code before running suite'; exit 1; }
if [[ -f "$OUTPUT_ROOT/CODE_COMMIT" ]]; then
  [[ "$(cat "$OUTPUT_ROOT/CODE_COMMIT")" == "$(git rev-parse HEAD)" ]] || {
    echo 'Refuse to mix different code revisions in one experiment'; exit 1;
  }
else
  git rev-parse HEAD > "$OUTPUT_ROOT/CODE_COMMIT"
  sha256sum "$G5_CKPT" > "$OUTPUT_ROOT/G5_CHECKPOINT_SHA256.txt"
fi
common=(--gpu 0 --seed 1029 --num_workers 8 --freeze_backbone
  --batch_size 48 --accumulation_steps 2 --dropout 0.0 --epochs 10
  --head_lr 3e-4 --lr 1e-5 --optim_type AdamW --weight_decay 0.05 --amp --head_fp32
  --normalize imagenet --load_size 256 --crop_size 224
  --zoom_crop_prob 0.5 --zoom_crop_scale 2.0 --no-zoom_crop_on_val --deterministic_val
  --val_threshold 0.90 --early_stop_patience 3 --dynamic_rank 16 --mlp_width 32
  --local_noise_radius 0.01 --log_every 250)

for specification in D1_dynamic_local:dynamic:1.0 C1_linear_continue:linear:0.0 A1_dynamic_no_local:dynamic:0.0 C2_residual_mlp:mlp:0.0; do
  IFS=: read -r run head local_weight <<< "$specification"
  run_root="$OUTPUT_ROOT/$run"
  mkdir -p "$run_root"
  stage=training
  if [[ ! -f "$run_root/training_complete.json" ]]; then
    # Do not silently retrain/overwrite a partially completed experiment.
    [[ ! -e "$run_root/TRAINING_STARTED" ]] || {
      echo "Partial training detected for $run: inspect logs before any restart"; exit 2;
    }
    touch "$run_root/TRAINING_STARTED"
    printf '%s\tTRAIN_START\t%s\n' "$(date --iso-8601=seconds)" "$run" >> "$STATUS_FILE"
    python main.py "${common[@]}" --classifier_type "$head" --local_weight "$local_weight" \
      --init_ckpt "$G5_CKPT" --output_root "$run_root" --test_sets stable_diffusion_v_1_4 \
      2>&1 | tee -a "$run_root/train_console.log"
    printf '%s\tTRAIN_COMPLETE\t%s\n' "$(date --iso-8601=seconds)" "$run" >> "$STATUS_FILE"
  fi
  checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' "$run_root/best_checkpoint.json")
  [[ -f "$checkpoint" ]]
  for benchmark in genimage chameleon; do
    stage="${benchmark}_eval"
    [[ -f "$run_root/${benchmark^^}_EVAL_FINISHED" ]] && continue
    if [[ "$benchmark" == genimage ]]; then
      root="$GENIMAGE_ROOT"
      subsets=("${TEST_SETS[@]}")
    else
      root="$CHAMELEON_ROOT"
      subsets=(.)
    fi
    printf '%s\tEVAL_START\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" "$benchmark" >> "$STATUS_FILE"
    python test.py "${common[@]}" --classifier_type "$head" --local_weight "$local_weight" \
      --ckpt_path "$checkpoint" --output_root "$run_root/${benchmark}_eval" \
      --test_root "$root" --test_sets "${subsets[@]}" \
      2>&1 | tee -a "$run_root/${benchmark}_eval_console.log"
    touch "$run_root/${benchmark^^}_EVAL_FINISHED"
    printf '%s\tEVAL_COMPLETE\t%s\t%s\n' "$(date --iso-8601=seconds)" "$run" "$benchmark" >> "$STATUS_FILE"
  done
done
stage=summary
python scripts/summarize_dynamic_head.py "$OUTPUT_ROOT"
touch "$OUTPUT_ROOT/SUITE_FINISHED"
printf '%s\tSUITE_COMPLETE\n' "$(date --iso-8601=seconds)" >> "$STATUS_FILE"
