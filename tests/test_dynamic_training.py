"""Fast CPU integration test for the actual training loop/checkpoint policy."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import main
from networks.dino_baseline import DINOv2Baseline
from networks.dynamic_head import DynamicLinearHead
from options import BaseOptions


class TinyFeatureModel(DINOv2Baseline):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = nn.Identity()
        self.classifier = DynamicLinearHead(nn.Sequential(nn.Dropout(0), nn.Linear(1024, 2)))
        self.head_fp32 = True


class TrainingIntegrationTests(unittest.TestCase):
    def test_checkpoint_trigger_early_stop_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            options = BaseOptions()
            args = options.initialize(argparse.ArgumentParser()).parse_args([])
            args.output_root = directory
            args.gpu = "-1"
            args.epochs = 10
            args.classifier_type = "dynamic"
            args.freeze_backbone = True
            args.local_weight = 1.0
            args.head_fp32 = True
            args.num_workers = 0
            args.batch_size = 4
            args.accumulation_steps = 2
            initial = nn.Sequential(nn.Dropout(0), nn.Linear(1024, 2))
            args.init_ckpt = str(Path(directory) / "g5.pt")
            torch.save({"classifier.1.weight": initial[1].weight,
                        "classifier.1.bias": initial[1].bias}, args.init_ckpt)
            data = TensorDataset(torch.randn(8, 1024), torch.arange(8) % 2)
            loader = DataLoader(data, batch_size=4)
            # First epoch below threshold: no selection test/checkpoint call.
            # Then improve at epoch 3, followed by three non-improvements.
            validation = [(v, 0, 0, 0) for v in [.89, .93, .94, .95, .95, .95]]
            with patch.object(main.BaseOptions, "parse", return_value=args), \
                 patch.object(main, "build_model", side_effect=lambda _: TinyFeatureModel()), \
                 patch.object(main, "create_dataloaders", return_value=(loader, loader, {"sdv14": loader})), \
                 patch.object(main, "validate", side_effect=validation), \
                 patch.object(main, "test_epoch", side_effect=[.92, .94, .93, .92, .91]) as selection:
                main.train()
            self.assertEqual(selection.call_count, 5)
            root = Path(directory)
            best = json.loads((root / "best_checkpoint.json").read_text())
            complete = json.loads((root / "training_complete.json").read_text())
            self.assertEqual(best["epoch"], 3)
            self.assertEqual(best["selection_test_acc"], .94)
            self.assertEqual(complete["completed_epochs"], 6)
            self.assertTrue(complete["early_stopped"])
            epochs = [json.loads(row) for row in (root / "epoch_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(epochs), 6)
            self.assertIsNone(epochs[0]["selection_test_acc"])
            model = TinyFeatureModel()
            model.load_state_dict(torch.load(best["path"], weights_only=True), strict=True)
            self.assertTrue(torch.isfinite(model(data.tensors[0])).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
