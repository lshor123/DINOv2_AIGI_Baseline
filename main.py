import csv
import datetime
import os
import random
import sys

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_loading import create_dataloaders
from networks.dino_baseline import DINOv2Baseline
from options import BaseOptions
from validation import validate


class EarlyStopping:
    """Retains the original code's test-accuracy checkpoint policy."""

    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score, model, save_path):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping.")
        else:
            self.best_score = score
            self.save_checkpoint(model, save_path)
            self.counter = 0

    @staticmethod
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)
        logger.info(f"Checkpoint saved to {path}")


def init_logger(log_dir, console_log_level="INFO"):
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=console_log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | {message}"
        ),
        colorize=True,
    )
    logger.add(
        os.path.join(log_dir, "debug_{time:YYYY-MM-DD}.log"),
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )


def set_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(gpu):
    if not gpu or gpu == "-1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. The current instance is in "
            "no-GPU mode; start a GPU instance before training, or use --gpu -1 "
            "only for CPU checks."
        )
    return torch.device(f"cuda:{gpu.split(',')[0]}")


def build_model(args):
    return DINOv2Baseline(
        backbone=args.backbone,
        repo_path=args.dino_repo,
        weights_path=args.dino_weights,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )


def get_optimizer(model, args):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr})
    param_groups.append({"params": head_params, "lr": args.head_lr})

    if args.optim_type == "Adam":
        return torch.optim.Adam(param_groups, betas=(0.9, 0.999))
    if args.optim_type == "SGD":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)
    return torch.optim.AdamW(
        param_groups, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )


def test_epoch(model, dl_test, device, args, time_str):
    rows = [["Parameters"]]
    rows.extend([[key, value] for key, value in sorted(vars(args).items())])
    rows.extend([[], ["Dataset", "Accuracy", "AP", "r_acc", "f_acc"]])
    accs, aps = [], []

    for test_type, loader in dl_test.items():
        acc, ap, r_acc, f_acc = validate(model, loader, device)[:4]
        rows.append([test_type, acc * 100, ap * 100, r_acc * 100, f_acc * 100])
        accs.append(acc)
        aps.append(ap)
        logger.info(
            f"({test_type:24}) acc: {acc*100:.4f}; ap: {ap*100:.4f}; "
            f"r_acc: {r_acc*100:.4f}; f_acc: {f_acc*100:.4f}"
        )

    mean_acc = float(np.mean(accs)) * 100
    mean_ap = float(np.mean(aps)) * 100
    logger.info(f"({'Mean':24}) acc: {mean_acc:.2f}; ap: {mean_ap:.2f}")
    rows.append(["MEAN", mean_acc, mean_ap])

    result_path = os.path.join(args.output_root, "results", f"test_{time_str}.csv")
    with open(result_path, "a", newline="", encoding="utf-8") as file_:
        csv.writer(file_).writerows(rows)
    return mean_acc / 100.0


def train():
    args = BaseOptions().parse()
    for name in ("checkpoints", "logs", "results", "runs"):
        os.makedirs(os.path.join(args.output_root, name), exist_ok=True)

    init_logger(os.path.join(args.output_root, "logs"))
    set_random_seed(args.seed)
    device = resolve_device(args.gpu)
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    writer = SummaryWriter(os.path.join(args.output_root, "runs", time_str))

    logger.info(f"Using device: {device}")
    model = build_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: total={total_params:,}, trainable={trainable_params:,}"
    )

    dl_train, dl_val, dl_test = create_dataloaders(args)
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    early_stopping = EarlyStopping(patience=args.early_stop_patience)
    criterion = nn.CrossEntropyLoss()
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = os.path.join(
        args.output_root, "checkpoints", f"genimage_DINOv2_ViTL14_{time_str}.pt"
    )
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for index, (inputs, labels) in enumerate(dl_train):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(inputs)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / args.accumulation_steps

            scaler.scale(loss).backward()
            should_step = (
                (index + 1) % args.accumulation_steps == 0
                or index + 1 == len(dl_train)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                writer.add_scalar("Loss/classification", raw_loss.item(), global_step)
                global_step += 1

            running_loss += raw_loss.item() * inputs.size(0)

        train_loss = running_loss / len(dl_train.dataset)
        logger.info(f"epoch[{epoch}] train_loss={train_loss:.6f}")

        val_acc = validate(model, dl_val, device)[0]
        logger.info(f"epoch[{epoch}] val_acc={100 * val_acc:.2f}%")
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        scheduler.step(val_acc)

        # Intentionally retain the original PPM-CLIP trigger and test-based
        # early-stopping/checkpoint selection as requested.
        if val_acc >= args.val_threshold:
            test_acc = test_epoch(model, dl_test, device, args, time_str)
            logger.info(f"epoch[{epoch}] test_acc={100 * test_acc:.2f}%")
            early_stopping(test_acc, model, checkpoint_path)
            if early_stopping.early_stop:
                break

    writer.close()


if __name__ == "__main__":
    train()
