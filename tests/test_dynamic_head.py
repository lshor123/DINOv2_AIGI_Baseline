import copy
import unittest

import torch
from torch import nn

from networks.dynamic_head import DynamicLinearHead, ResidualMLPHead, sample_feature_noise


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1029)
        self.base = nn.Sequential(nn.Dropout(0), nn.Linear(1024, 2))
        self.z = torch.randn(8, 1024)

    def test_initialization_and_parameter_counts(self):
        for cls, count in ((DynamicLinearHead, 34818), (ResidualMLPHead, 34850)):
            rng = torch.get_rng_state().clone()
            head = cls(copy.deepcopy(self.base))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in head.parameters()), count)
            self.assertTrue(torch.equal(head(self.z), self.base(self.z)))
            self.assertTrue(torch.equal(head(self.z).softmax(-1), self.base(self.z).softmax(-1)))

    def test_dynamic_score_formula(self):
        head = DynamicLinearHead(copy.deepcopy(self.base))
        nn.init.normal_(head.up.weight, std=0.03)
        logits = head(self.z)
        bias = head[1].bias[1] - head[1].bias[0]
        expected = (head.effective_weight(self.z) * self.z).sum(-1) + bias
        torch.testing.assert_close(logits[:, 1] - logits[:, 0], expected)

    def test_first_step_and_subsequent_gradient(self):
        head = DynamicLinearHead(copy.deepcopy(self.base))
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        labels = torch.arange(8) % 2
        nn.functional.cross_entropy(head(self.z), labels).backward()
        self.assertGreater(head.up.weight.grad.abs().sum().item(), 0)
        self.assertEqual(head.down.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        nn.functional.cross_entropy(head(self.z), labels).backward()
        self.assertGreater(head.down.weight.grad.abs().sum().item(), 0)

    def test_local_identity_and_gradients(self):
        head = DynamicLinearHead(copy.deepcopy(self.base))
        generator = torch.Generator().manual_seed(22)
        eps = sample_feature_noise(self.z, 0.01, generator)
        self.assertEqual(head.neighborhood_loss(self.z, eps).item(), 0)
        nn.init.normal_(head.up.weight, std=0.1)
        stable = head.neighborhood_loss(self.z, eps)
        logits = head(self.z)
        neighbor_logits = head(self.z + eps)
        direct_residual = ((neighbor_logits[:, 1] - neighbor_logits[:, 0])
                           - (logits[:, 1] - logits[:, 0])
                           - (head.effective_weight(self.z) * eps).sum(-1))
        direct = (direct_residual.square() / (eps.square().sum(-1) + 1e-8)).mean()
        torch.testing.assert_close(stable, direct, rtol=5e-4, atol=1e-8)
        parameters = [head.up.weight, head.down.weight]
        grads_stable = torch.autograd.grad(stable, parameters)
        grads_direct = torch.autograd.grad(direct, parameters)
        for lhs, rhs in zip(grads_stable, grads_direct):
            self.assertTrue(torch.isfinite(lhs).all())
            self.assertGreater(lhs.abs().sum().item(), 0)
            torch.testing.assert_close(lhs, rhs, rtol=0.015, atol=3e-6)

    def test_noise_radius_and_rng_isolation(self):
        global_rng = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(777)
        eps = sample_feature_noise(self.z, 0.01, generator)
        torch.testing.assert_close(eps.norm(dim=-1), 0.01 * self.z.norm(dim=-1))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))

    def test_strict_checkpoint_roundtrip(self):
        for cls in (DynamicLinearHead, ResidualMLPHead):
            original = cls(copy.deepcopy(self.base))
            for parameter in original.parameters():
                with torch.no_grad():
                    parameter.add_(0.01)
            restored = cls(copy.deepcopy(self.base))
            restored.load_state_dict(original.state_dict(), strict=True)
            self.assertTrue(torch.equal(original(self.z), restored(self.z)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
