import csv
import datetime
import os

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score

from data_loading import create_test_dataloaders
from main import build_model, init_logger, resolve_device, set_random_seed
from options import BaseOptions
from validation import validate


def test_epoch(model, loaders, device, args, time_str):
    rows = [["Parameters"]]
    rows.extend([[key, value] for key, value in sorted(vars(args).items())])
    rows.extend([[], ["Dataset", "Accuracy", "AUC", "AP", "r_acc", "f_acc"]])
    accs, aucs, aps = [], [], []

    for test_type, loader in loaders.items():
        acc, ap, r_acc, f_acc, y_true, y_pred = validate(model, loader, device)
        auc = roc_auc_score(y_true, y_pred)
        rows.append(
            [test_type, acc * 100, auc * 100, ap * 100, r_acc * 100, f_acc * 100]
        )
        accs.append(acc)
        aucs.append(auc)
        aps.append(ap)
        logger.info(
            f"({test_type:24}) Acc={acc*100:.2f}; AUC={auc*100:.2f}; "
            f"AP={ap*100:.2f}; R_Acc={r_acc*100:.2f}; F_Acc={f_acc*100:.2f}"
        )

    means = [np.mean(accs) * 100, np.mean(aucs) * 100, np.mean(aps) * 100]
    rows.append(["MEAN", *means])
    logger.info(f"Mean Acc={means[0]:.2f}; AUC={means[1]:.2f}; AP={means[2]:.2f}")

    result_path = os.path.join(args.output_root, "results", f"eval_{time_str}.csv")
    with open(result_path, "w", newline="", encoding="utf-8") as file_:
        csv.writer(file_).writerows(rows)
    logger.info(f"Results saved to {result_path}")


def main():
    args = BaseOptions().parse()
    if not args.ckpt_path:
        raise ValueError("Testing requires --ckpt_path /path/to/checkpoint.pt")
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    for name in ("logs", "results"):
        os.makedirs(os.path.join(args.output_root, name), exist_ok=True)
    init_logger(os.path.join(args.output_root, "logs"))
    set_random_seed(args.seed)
    device = resolve_device(args.gpu)

    loaders = create_test_dataloaders(args)
    model = build_model(args)
    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    test_epoch(
        model,
        loaders,
        device,
        args,
        datetime.datetime.now().strftime("%m%d_%H%M"),
    )


if __name__ == "__main__":
    main()
