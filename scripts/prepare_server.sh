#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/root/PPM_CLIP_DINO_baseline}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
VENV_ROOT="${VENV_ROOT:-${DATA_ROOT}/envs/dino_baseline}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-${DATA_ROOT}/pretrained}"
DINOV2_COMMIT="${DINOV2_COMMIT:-7764ea0f912e53c92e82eb78a2a1631e92725fc8}"

DATA_ROOT="$(realpath -m "${DATA_ROOT}")"
PRETRAINED_ROOT="$(realpath -m "${PRETRAINED_ROOT}")"
case "${PRETRAINED_ROOT}/" in
  "${DATA_ROOT}/"*) ;;
  *) echo "PRETRAINED_ROOT must be inside DATA_ROOT" >&2; exit 1 ;;
esac

mkdir -p \
  "${DATA_ROOT}/envs" \
  "${DATA_ROOT}/datasets/GenImage" \
  "${DATA_ROOT}/outputs/dinov2_vitl14_baseline" \
  "${DATA_ROOT}/cache/huggingface" \
  "${DATA_ROOT}/cache/torch" \
  "${PRETRAINED_ROOT}"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_ROOT}"
fi

"${VENV_ROOT}/bin/python" -m pip install --upgrade pip
"${VENV_ROOT}/bin/python" -m pip install -r "${CODE_ROOT}/requirements.txt"

if [[ ! -f "${PRETRAINED_ROOT}/dinov2/hubconf.py" ]]; then
  rm -rf "${PRETRAINED_ROOT}/dinov2" "${PRETRAINED_ROOT}/dinov2-${DINOV2_COMMIT}"
  wget -c "https://codeload.github.com/facebookresearch/dinov2/zip/${DINOV2_COMMIT}" \
    -O "${PRETRAINED_ROOT}/dinov2-source.zip"
  unzip -q "${PRETRAINED_ROOT}/dinov2-source.zip" -d "${PRETRAINED_ROOT}"
  mv "${PRETRAINED_ROOT}/dinov2-${DINOV2_COMMIT}" "${PRETRAINED_ROOT}/dinov2"
  rm -f "${PRETRAINED_ROOT}/dinov2-source.zip"
fi
printf '%s\n' "${DINOV2_COMMIT}" > "${PRETRAINED_ROOT}/dinov2/COMMIT"

WEIGHTS="${PRETRAINED_ROOT}/dinov2_vitl14_pretrain.pth"
WEIGHTS_SHA256="d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
if ! printf '%s  %s\n' "${WEIGHTS_SHA256}" "${WEIGHTS}" | sha256sum -c -; then
  wget -c \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
    -O "${WEIGHTS}"
  printf '%s  %s\n' "${WEIGHTS_SHA256}" "${WEIGHTS}" | sha256sum -c -
fi

cat > "${DATA_ROOT}/activate_dino.sh" <<EOF
#!/usr/bin/env bash
source "${VENV_ROOT}/bin/activate"
export PYTHONPATH="${CODE_ROOT}:\${PYTHONPATH:-}"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export GENIMAGE_ROOT="${DATA_ROOT}/datasets/GenImage"
export GENIMAGE_SDV14_ROOT="${DATA_ROOT}/datasets/GenImage/stable_diffusion_v_1_4"
export DINOV2_REPO="${PRETRAINED_ROOT}/dinov2"
export DINOV2_WEIGHTS="${WEIGHTS}"
export DINO_OUTPUT_ROOT="${DATA_ROOT}/outputs/dinov2_vitl14_baseline"
EOF
chmod +x "${DATA_ROOT}/activate_dino.sh"

"${VENV_ROOT}/bin/python" - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA currently available:", torch.cuda.is_available())
PY

echo "Environment prepared. Activate with: source ${DATA_ROOT}/activate_dino.sh"
