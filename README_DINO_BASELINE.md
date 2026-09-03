# DINOv2 ViT-L/14 baseline

This branch replaces the complete PPM-CLIP model with a pure image-only
baseline:

```text
224 x 224 RGB image
  -> DINOv2 ViT-L/14 pretrained backbone
  -> 1024-dimensional CLS feature
  -> Linear(1024, 2)
  -> real/fake logits
```

The baseline contains no CLIP text encoder, prompt repository, PPM/flow,
ICD/IGD/ISD, LoRA, PWCL, DCT/frequency branch, or auxiliary loss. Training
uses only binary cross-entropy in its two-logit `CrossEntropyLoss` form.

## Server layout

```text
/root/PPM_CLIP_DINO_baseline/                 code (system disk)
/root/autodl-tmp/envs/dino_baseline/          Python environment
/root/autodl-tmp/pretrained/                  DINOv2 repo and weights
/root/autodl-tmp/datasets/GenImage/           datasets
/root/autodl-tmp/outputs/dinov2_vitl14_baseline/ checkpoints/logs/results
```

Prepare dependencies and weights:

```bash
bash /root/PPM_CLIP_DINO_baseline/scripts/prepare_server.sh
```

Download and extract GenImage Stable Diffusion v1.4:

```bash
bash /root/PPM_CLIP_DINO_baseline/scripts/download_sdv14.sh
```

Verify the environment, weights and dataset before renting the GPU:

```bash
source /root/autodl-tmp/activate_dino.sh
cd /root/PPM_CLIP_DINO_baseline
python scripts/verify_setup.py
```

After switching to GPU mode, run the one-batch backbone check once:

```bash
python scripts/verify_setup.py --full_model_check
```

Start training after changing the instance to GPU mode:

```bash
screen -S dino_train
bash /root/PPM_CLIP_DINO_baseline/scripts/train_sdv14.sh
```

Detach from screen with `Ctrl+A`, then `D`. Reattach with
`screen -r dino_train`.

The physical batch size is 4 and gradients are accumulated over 4 steps, so
the effective batch size is 16. AMP is enabled by default. If CUDA runs out of
memory, reduce `--batch_size` to 2 and increase `--accumulation_steps` to 8.

## Testing later

GenImage example:

```bash
source /root/autodl-tmp/activate_dino.sh
cd /root/PPM_CLIP_DINO_baseline
python test.py \
  --gpu 0 \
  --ckpt_path /root/autodl-tmp/outputs/dinov2_vitl14_baseline/checkpoints/FILE.pt \
  --test_root /root/autodl-tmp/datasets/GenImage \
  --test_sets Midjourney stable_diffusion_v_1_4 stable_diffusion_v_1_5 ADM glide wukong VQDM BigGAN
```

For Chameleon, point `--test_root` to the directory whose recursive contents
contain `real/fake` or `0_real/1_fake` class folders, and use `--test_sets .`.
