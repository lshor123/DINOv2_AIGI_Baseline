import io
import os
from pathlib import Path

import torch
import torchvision
from loguru import logger
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REAL_DIR_NAMES = {"0_real", "nature", "real"}
FAKE_DIR_NAMES = {"1_fake", "ai", "fake"}


class ForenSynths(Dataset):
    """Recursive real/fake image dataset supporting PPM-CLIP and GenImage names."""

    def __init__(self, root_dir, transform):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.transform = transform
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        self.data = []
        class_counts = [0, 0]
        for root, _, files in os.walk(self.root_dir):
            label = self._label_from_path(root)
            if label is None:
                continue
            for filename in files:
                if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                self.data.append((os.path.join(root, filename), label))
                class_counts[label] += 1

        if not self.data:
            raise RuntimeError(
                f"No labeled images found below {self.root_dir}. Expected class "
                "directories named nature/ai or 0_real/1_fake."
            )
        if 0 in class_counts:
            raise RuntimeError(
                f"Both classes are required below {self.root_dir}; "
                f"found real={class_counts[0]}, fake={class_counts[1]}."
            )
        logger.info(
            f"Loaded {self.root_dir}: real={class_counts[0]}, "
            f"fake={class_counts[1]}, total={len(self.data)}"
        )

    @staticmethod
    def _label_from_path(path):
        parts = {part.lower() for part in Path(path).parts}
        if parts & REAL_DIR_NAMES:
            return 0
        if parts & FAKE_DIR_NAMES:
            return 1
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        try:
            with Image.open(img_path) as source:
                image = source.convert("RGB")
            return self.transform(image), label
        except Exception as exc:
            logger.error(f"Error loading image {img_path}: {exc}")
            return torch.zeros(3, 224, 224), label


def resolve_split_root(root_dir, split):
    """Find split in either root/split or root/<single archive folder>/split."""
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    if os.path.basename(root_dir).lower() == split.lower() and os.path.isdir(root_dir):
        return root_dir

    direct = os.path.join(root_dir, split)
    if os.path.isdir(direct):
        return direct

    if os.path.isdir(root_dir):
        candidates = []
        for child in os.scandir(root_dir):
            if child.is_dir():
                candidate = os.path.join(child.path, split)
                if os.path.isdir(candidate):
                    candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple {split!r} directories found below {root_dir}: {candidates}. "
                "Pass an explicit --train_root/--val_root."
            )

    raise FileNotFoundError(
        f"Could not find a {split!r} directory in {root_dir} or one level below it."
    )


def judge_img(img, load_size):
    width, height = img.size
    if width < load_size or height < load_size:
        img = torchvision.transforms.Resize(
            (load_size, load_size), interpolation=InterpolationMode.BILINEAR
        )(img)
    return img


def jpeg_compression(img, quality=50):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class RandomZoomCrop:
    """RandomCrop with an optional image-space zoom before the crop.

    With probability ``probability``, resize an H x W image to
    ``scale * H x scale * W`` and then take the same fixed-size random crop
    used by the original baseline. With probability ``1 - probability``, this
    is exactly the original RandomCrop operation.
    """

    def __init__(self, size, probability=0.5, scale=2.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if scale < 1.0:
            raise ValueError("scale must be at least 1.0")
        self.size = size
        self.probability = probability
        self.scale = scale
        self.random_crop = transforms.RandomCrop(size)

    def __call__(self, image):
        if torch.rand(()) < self.probability:
            width, height = image.size
            resized_width = max(1, round(width * self.scale))
            resized_height = max(1, round(height * self.scale))
            image = TF.resize(
                image,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return self.random_crop(image)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(size={self.size}, "
            f"probability={self.probability}, scale={self.scale})"
        )


def train_augment(
    normalize,
    load_size,
    crop_size,
    zoom_crop_prob=0.5,
    zoom_crop_scale=2.0,
):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: judge_img(img, load_size)),
            RandomZoomCrop(
                crop_size,
                probability=zoom_crop_prob,
                scale=zoom_crop_scale,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[normalize], std=STD[normalize]),
        ]
    )


def validation_augment(
    normalize,
    load_size,
    crop_size,
    zoom_crop_on_val=False,
    zoom_crop_prob=0.5,
    zoom_crop_scale=2.0,
):
    crop = (
        RandomZoomCrop(
            crop_size,
            probability=zoom_crop_prob,
            scale=zoom_crop_scale,
        )
        if zoom_crop_on_val
        else transforms.RandomCrop(crop_size)
    )
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: judge_img(img, load_size)),
            crop,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[normalize], std=STD[normalize]),
        ]
    )


def test_augment(normalize, load_size, crop_size):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: judge_img(img, load_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[normalize], std=STD[normalize]),
        ]
    )


def _loader_kwargs(args):
    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return kwargs


def _training_roots(args):
    train_root = args.train_root or resolve_split_root(args.sdv14_root, "train")
    val_root = args.val_root or resolve_split_root(args.sdv14_root, "val")
    return train_root, val_root


def create_dataloaders(args):
    train_root, val_root = _training_roots(args)
    loader_kwargs = _loader_kwargs(args)

    dl_train = torch.utils.data.DataLoader(
        ForenSynths(
            train_root,
            train_augment(
                args.normalize,
                args.load_size,
                args.crop_size,
                args.zoom_crop_prob,
                args.zoom_crop_scale,
            ),
        ),
        shuffle=True,
        **loader_kwargs,
    )
    # Retain the original random crop/flip validation pipeline unless the
    # experiment explicitly enables zoom augmentation or requests the exact
    # deterministic test transform for validation.
    val_transform = (
        test_augment(args.normalize, args.load_size, args.crop_size)
        if args.deterministic_val
        else validation_augment(
            args.normalize,
            args.load_size,
            args.crop_size,
            args.zoom_crop_on_val,
            args.zoom_crop_prob,
            args.zoom_crop_scale,
        )
    )
    dl_val = torch.utils.data.DataLoader(
        ForenSynths(val_root, val_transform),
        shuffle=False,
        **loader_kwargs,
    )
    return dl_train, dl_val, create_test_dataloaders(args)


def create_test_dataloaders(args):
    loader_kwargs = _loader_kwargs(args)
    transform = test_augment(args.normalize, args.load_size, args.crop_size)
    loaders = {}
    for name in args.test_sets:
        generator_root = args.test_root if name == "." else os.path.join(args.test_root, name)
        try:
            test_dir = resolve_split_root(generator_root, "val")
        except FileNotFoundError:
            # Chameleon-style benchmarks may place real/fake folders directly
            # below the requested root without a separate val directory.
            test_dir = generator_root
        loaders[name] = torch.utils.data.DataLoader(
            ForenSynths(test_dir, transform),
            shuffle=False,
            **loader_kwargs,
        )
    return loaders
