"""GPU smoke check on real TRAIN images; never writes to the G5 checkpoint."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from data_loading import create_dataloaders
from main import build_model, get_optimizer, resolve_device, set_random_seed
from networks.dynamic_head import sample_feature_noise
from options import BaseOptions


args = BaseOptions().parse()
assert args.classifier_type == "dynamic" and args.freeze_backbone and args.init_ckpt
assert args.dropout == 0 and args.deterministic_val and not args.zoom_crop_on_val
set_random_seed(args.seed)
device = resolve_device(args.gpu)
model = build_model(args)
model.load_g5_initialization(args.init_ckpt)
model.to(device).train()
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 34818
backbone_sample = next(model.backbone.parameters()).detach().clone()
optimizer = get_optimizer(model, args)
generator = torch.Generator(device=device).manual_seed(args.seed + 100003)
train_loader, val_loader, _ = create_dataloaders(args)
assert len(train_loader.dataset) == 324000
assert len(val_loader.dataset) == 12000
iterator = iter(train_loader)
scaler = torch.amp.GradScaler("cuda", enabled=True)
records = []
optimizer.zero_grad(set_to_none=True)
for index in range(4):  # Two optimizer steps with the requested accumulation=2.
    images, labels = next(iterator)
    images, labels = images.to(device), labels.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        features = model.forward_features(images)
        logits = model.forward_head(features)
    with torch.autocast("cuda", enabled=False):
        features = features.float()
        if index == 0:
            reference_logits = model.classifier.base_logits(features)
            torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
        noise = sample_feature_noise(features, args.local_noise_radius, generator)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        local = model.classifier.neighborhood_loss(features, noise)
        total = ce + args.local_weight * local
    assert torch.isfinite(total)
    scaler.scale(total / args.accumulation_steps).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.classifier.parameters())
    if (index + 1) % args.accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    records.append({"batch": index + 1, "ce": ce.item(), "local": local.item(),
                    "features": list(features.shape), "logits": list(logits.shape)})
assert torch.equal(backbone_sample, next(model.backbone.parameters()))
assert model.classifier.up.weight.abs().sum().item() > 0
assert model.classifier.down.weight.isfinite().all()
model.eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    output = model(images)
assert output.shape == (args.batch_size, 2) and output.dtype == torch.float32
print(json.dumps({"status": "PASS", "initial_logits_equal_G5": True,
                  "backbone_frozen": True, "trainable_parameters": 34818,
                  "peak_gpu_memory_MiB": torch.cuda.max_memory_allocated() / 2**20,
                  "batches": records}, indent=2))
