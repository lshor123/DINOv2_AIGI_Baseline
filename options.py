import argparse
import os

from loguru import logger


DEFAULT_GENIMAGE_ROOT = os.environ.get(
    "GENIMAGE_ROOT", "/root/autodl-tmp/datasets/GenImage"
)
DEFAULT_SDV14_ROOT = os.environ.get(
    "GENIMAGE_SDV14_ROOT",
    os.path.join(DEFAULT_GENIMAGE_ROOT, "stable_diffusion_v_1_4"),
)
DEFAULT_DINO_REPO = os.environ.get(
    "DINOV2_REPO", "/root/autodl-tmp/pretrained/dinov2"
)
DEFAULT_DINO_WEIGHTS = os.environ.get(
    "DINOV2_WEIGHTS",
    "/root/autodl-tmp/pretrained/dinov2_vitl14_pretrain.pth",
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "DINO_OUTPUT_ROOT", "/root/autodl-tmp/outputs/dinov2_vitl14_baseline"
)


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        # Runtime and reproducibility.
        parser.add_argument("--gpu", type=str, default="0", help="GPU id; use -1 for CPU")
        parser.add_argument("--seed", type=int, default=1029)
        parser.add_argument("--num_workers", type=int, default=8)

        # Pure DINOv2 baseline: no CLIP prompts, LoRA, PPM, PWCL or DCT branch.
        parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
        parser.add_argument("--dino_repo", type=str, default=DEFAULT_DINO_REPO)
        parser.add_argument("--dino_weights", type=str, default=DEFAULT_DINO_WEIGHTS)
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument(
            "--freeze_backbone",
            action="store_true",
            help="Train only the linear classifier; default is full fine-tuning",
        )

        # GenImage SD v1.4 data. The loader discovers train/val below sdv14_root.
        parser.add_argument("--dataset", type=str, default="genimage")
        parser.add_argument("--sdv14_root", type=str, default=DEFAULT_SDV14_ROOT)
        parser.add_argument("--train_root", type=str, default="")
        parser.add_argument("--val_root", type=str, default="")
        parser.add_argument("--test_root", type=str, default=DEFAULT_GENIMAGE_ROOT)
        parser.add_argument(
            "--test_sets",
            nargs="+",
            default=["stable_diffusion_v_1_4"],
            help="Subdirectories under test_root; use '.' for test_root itself",
        )
        parser.add_argument("--normalize", choices=["imagenet", "clip"], default="imagenet")
        parser.add_argument("--load_size", type=int, default=256)
        parser.add_argument("--crop_size", type=int, default=224)
        parser.add_argument(
            "--zoom_crop_prob",
            type=float,
            default=0.5,
            help="Probability of resizing both image dimensions before RandomCrop",
        )
        parser.add_argument(
            "--zoom_crop_scale",
            type=float,
            default=2.0,
            help="Image enlargement factor used by RandomZoomCrop",
        )
        parser.add_argument(
            "--zoom_crop_on_val",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Apply RandomZoomCrop to validation as well as training",
        )
        parser.add_argument(
            "--deterministic_val",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use the exact test transform for validation: size check plus CenterCrop",
        )

        # Requested defaults for the RTX 3090 DINOv2 ViT-L/14 experiments.
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=48)
        parser.add_argument("--accumulation_steps", type=int, default=2)
        parser.add_argument("--optim_type", choices=["Adam", "AdamW", "SGD"], default="AdamW")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--head_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

        # Keep the original repository's validation/test checkpoint policy.
        parser.add_argument("--val_threshold", type=float, default=0.90)
        parser.add_argument("--early_stop_patience", type=int, default=3)
        parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
        parser.add_argument("--ckpt_path", type=str, default="")
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            self.parser = self.initialize(self.parser)
        return self.parser.parse_args()

    def print_options(self, opt):
        lines = ["----------------- Options ---------------"]
        for key, value in sorted(vars(opt).items()):
            default = self.parser.get_default(key)
            comment = f"\t[default: {default}]" if value != default else ""
            lines.append(f"{key:>25}: {str(value):<30}{comment}")
        lines.append("----------------- End -------------------")
        logger.info("\n".join(lines))

    def parse(self, print_options=True):
        opt = self.gather_options()
        if not 0.0 <= opt.zoom_crop_prob <= 1.0:
            self.parser.error("--zoom_crop_prob must be between 0 and 1")
        if opt.zoom_crop_scale < 1.0:
            self.parser.error("--zoom_crop_scale must be at least 1.0")
        if opt.deterministic_val and opt.zoom_crop_on_val:
            self.parser.error(
                "--deterministic_val and --zoom_crop_on_val cannot be enabled together"
            )
        self.opt = opt
        if print_options:
            self.print_options(opt)
        return opt
