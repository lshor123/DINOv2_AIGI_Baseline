#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
DATASET_ROOT="${DATA_ROOT}/datasets/GenImage"
TEST_ROOT="${DATASET_ROOT}/genimage_test"
ZIP_PATH="${DATASET_ROOT}/genimage_test.zip"
LOG_ROOT="${DATA_ROOT}/logs"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
EXPECTED_ZIP_BYTES=25408572820
MIN_FREE_KIB=$((55 * 1024 * 1024))

mkdir -p "${DATASET_ROOT}" "${TEST_ROOT}" "${LOG_ROOT}"

if ! command -v 7z >/dev/null 2>&1 || ! command -v aria2c >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y p7zip-full aria2
fi

AVAILABLE_KIB=$(df --output=avail "${DATA_ROOT}" | tail -n 1 | tr -d ' ')
if [[ "${AVAILABLE_KIB}" -lt "${MIN_FREE_KIB}" && ! -f "${ZIP_PATH}" ]]; then
  echo "At least 55 GiB of free data-disk space is required." >&2
  exit 1
fi

URL="${HF_ENDPOINT}/datasets/jzousz/GenImage/resolve/main/genimage_test.zip?download=true"
aria2c \
  --dir="${DATASET_ROOT}" \
  --out="genimage_test.zip" \
  --continue=true \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=16M \
  --file-allocation=none \
  --auto-file-renaming=false \
  --summary-interval=60 \
  "${URL}"

ACTUAL_ZIP_BYTES=$(stat -c '%s' "${ZIP_PATH}")
if [[ "${ACTUAL_ZIP_BYTES}" -ne "${EXPECTED_ZIP_BYTES}" ]]; then
  echo "Unexpected ZIP size: ${ACTUAL_ZIP_BYTES}; expected ${EXPECTED_ZIP_BYTES}." >&2
  exit 1
fi

7z x "${ZIP_PATH}" -o"${TEST_ROOT}" -y

declare -A EXPECTED_COUNTS=(
  [adm_imagenet]=6000
  [biggan_imagenet]=6000
  [glide_imagenet]=6000
  [midjourney_imagenet]=6000
  [sdv4_imagenet]=6000
  [sdv5_imagenet]=8000
  [vqdm_imagenet]=6000
  [wukong_imagenet]=6000
)

TOTAL_AI=0
TOTAL_NATURE=0
for subset in adm_imagenet biggan_imagenet glide_imagenet midjourney_imagenet \
              sdv4_imagenet sdv5_imagenet vqdm_imagenet wukong_imagenet; do
  AI_DIR="${TEST_ROOT}/test/${subset}/ai"
  NATURE_DIR="${TEST_ROOT}/test/${subset}/nature"
  AI_COUNT=$(find "${AI_DIR}" -maxdepth 1 -type f | wc -l)
  NATURE_COUNT=$(find "${NATURE_DIR}" -maxdepth 1 -type f | wc -l)
  EXPECTED="${EXPECTED_COUNTS[${subset}]}"
  printf '%s ai=%s nature=%s\n' "${subset}" "${AI_COUNT}" "${NATURE_COUNT}"
  if [[ "${AI_COUNT}" -ne "${EXPECTED}" || "${NATURE_COUNT}" -ne "${EXPECTED}" ]]; then
    echo "Dataset verification failed for ${subset}; ZIP was retained." >&2
    exit 1
  fi
  TOTAL_AI=$((TOTAL_AI + AI_COUNT))
  TOTAL_NATURE=$((TOTAL_NATURE + NATURE_COUNT))
done

if [[ "${TOTAL_AI}" -ne 50000 || "${TOTAL_NATURE}" -ne 50000 ]]; then
  echo "Unexpected total counts; ZIP was retained." >&2
  exit 1
fi

cat > "${TEST_ROOT}/DATASET_SOURCE.txt" <<EOF
repository=https://huggingface.co/datasets/jzousz/GenImage
revision=71c983e6262684bc2c6b6af99582e8f568c259a5
archive=genimage_test.zip
archive_bytes=${EXPECTED_ZIP_BYTES}
total_ai=${TOTAL_AI}
total_nature=${TOTAL_NATURE}
EOF

rm -f -- "${ZIP_PATH}" "${ZIP_PATH}.aria2"
echo "GenImage test set is ready at ${TEST_ROOT}/test"
