import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPECTED_WEIGHT_BYTES = 1_217_586_395
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_count(path):
    return sum(
        Path(filename).suffix.lower() in IMAGE_SUFFIXES
        for root, _, files in os.walk(path)
        for filename in files
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/autodl-tmp")
    parser.add_argument("--full_model_check", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    repo = data_root / "pretrained" / "dinov2"
    weights = data_root / "pretrained" / "dinov2_vitl14_pretrain.pth"
    subset = data_root / "datasets" / "GenImage" / "stable_diffusion_v_1_4"

    report = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "dino_repo": str(repo),
        "weights": str(weights),
        "sdv14_root": str(subset),
    }

    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing DINOv2 source: {repo}")
    if not weights.is_file():
        raise FileNotFoundError(f"Missing DINOv2 weights: {weights}")
    report["weight_bytes"] = weights.stat().st_size
    if report["weight_bytes"] != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"Incomplete DINOv2 weights: {report['weight_bytes']} != {EXPECTED_WEIGHT_BYTES}"
        )

    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or "patch_embed.proj.weight" not in state:
        raise RuntimeError("The DINOv2 checkpoint does not contain the expected backbone keys")
    report["weight_tensor_count"] = len(state)
    del state

    split_candidates = list(subset.glob("*/train"))
    if (subset / "train").is_dir():
        split_candidates.append(subset / "train")
    if len(split_candidates) != 1:
        raise RuntimeError(f"Could not uniquely locate SD v1.4 train split: {split_candidates}")
    train_root = split_candidates[0]
    val_root = train_root.parent / "val"

    counts = {}
    for split_name, split_root in (("train", train_root), ("val", val_root)):
        for class_name in ("nature", "ai"):
            class_root = split_root / class_name
            counts[f"{split_name}_{class_name}"] = image_count(class_root)
            if counts[f"{split_name}_{class_name}"] == 0:
                raise RuntimeError(f"No images found in {class_root}")
    report["image_counts"] = counts

    if args.full_model_check:
        if not torch.cuda.is_available():
            raise RuntimeError("--full_model_check requires a GPU instance")
        from networks.dino_baseline import DINOv2Baseline

        model = DINOv2Baseline(repo_path=str(repo), weights_path=str(weights)).cuda().eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model(torch.zeros(1, 3, 224, 224, device="cuda"))
        report["full_model_output_shape"] = list(output.shape)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("SETUP_CHECK_PASSED")


if __name__ == "__main__":
    main()
