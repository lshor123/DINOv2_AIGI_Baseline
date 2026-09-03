#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
VENV_ROOT="${VENV_ROOT:-${DATA_ROOT}/envs/dino_baseline}"
DATASET_ROOT="${DATA_ROOT}/datasets/GenImage"
ARCHIVE_ROOT="${DATASET_ROOT}/stable_diffusion_v_1_4"
LOG_ROOT="${DATA_ROOT}/logs"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${ARCHIVE_ROOT}" "${LOG_ROOT}"

if ! command -v 7z >/dev/null 2>&1 || ! command -v aria2c >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y p7zip-full aria2
fi

URL_LIST="${DATA_ROOT}/download_sdv14.urls"
: > "${URL_LIST}"
for index in $(seq -w 1 29); do
  filename="imagenet_ai_0419_sdv4.z${index}"
  url="${HF_ENDPOINT}/datasets/jzousz/GenImage/resolve/main/stable_diffusion_v_1_4/${filename}?download=true"
  printf '%s\n  dir=%s\n  out=%s\n' "${url}" "${ARCHIVE_ROOT}" "${filename}" >> "${URL_LIST}"
done
filename="imagenet_ai_0419_sdv4.zip"
url="${HF_ENDPOINT}/datasets/jzousz/GenImage/resolve/main/stable_diffusion_v_1_4/${filename}?download=true"
printf '%s\n  dir=%s\n  out=%s\n' "${url}" "${ARCHIVE_ROOT}" "${filename}" >> "${URL_LIST}"

aria2c \
  --input-file="${URL_LIST}" \
  --continue=true \
  --max-concurrent-downloads=8 \
  --max-connection-per-server=4 \
  --split=4 \
  --min-split-size=16M \
  --file-allocation=none \
  --auto-file-renaming=false \
  --summary-interval=60
rm -f "${URL_LIST}"

7z x "${ARCHIVE_ROOT}/imagenet_ai_0419_sdv4.zip" \
  -o"${ARCHIVE_ROOT}" -y

TRAIN_REAL=$(find "${ARCHIVE_ROOT}" -type f -path '*/train/nature/*' | wc -l)
TRAIN_FAKE=$(find "${ARCHIVE_ROOT}" -type f -path '*/train/ai/*' | wc -l)
VAL_REAL=$(find "${ARCHIVE_ROOT}" -type f -path '*/val/nature/*' | wc -l)
VAL_FAKE=$(find "${ARCHIVE_ROOT}" -type f -path '*/val/ai/*' | wc -l)

printf 'train nature=%s ai=%s; val nature=%s ai=%s\n' \
  "${TRAIN_REAL}" "${TRAIN_FAKE}" "${VAL_REAL}" "${VAL_FAKE}"

if [[ "${TRAIN_REAL}" -eq 0 || "${TRAIN_FAKE}" -eq 0 || \
      "${VAL_REAL}" -eq 0 || "${VAL_FAKE}" -eq 0 ]]; then
  echo "Dataset verification failed; archive files were retained." >&2
  exit 1
fi

cat > "${ARCHIVE_ROOT}/DATASET_SOURCE.txt" <<EOF
repository=https://huggingface.co/datasets/jzousz/GenImage
revision=71c983e6262684bc2c6b6af99582e8f568c259a5
subset=stable_diffusion_v_1_4
train_nature=${TRAIN_REAL}
train_ai=${TRAIN_FAKE}
val_nature=${VAL_REAL}
val_ai=${VAL_FAKE}
EOF

# The 200 GB data disk cannot permanently hold both the 96.4 GB multipart
# archive and the extracted images. Remove archives only after verification.
rm -f "${ARCHIVE_ROOT}"/imagenet_ai_0419_sdv4.z* \
      "${ARCHIVE_ROOT}/imagenet_ai_0419_sdv4.zip"

echo "SD v1.4 dataset is ready at ${ARCHIVE_ROOT}"
