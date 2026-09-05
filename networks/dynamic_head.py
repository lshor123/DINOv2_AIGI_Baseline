"""Residual classification heads with an exact G5 linear-head initialization."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualHead(nn.Sequential):
    """Keep G5's ``classifier.1.{weight,bias}`` checkpoint keys unchanged."""

    def __init__(self, linear_head):
        super().__init__(*list(linear_head.children()))

    @staticmethod
    def add_score(logits, score):
        # Preserve the old two logits exactly when the residual is zero.
        return logits + torch.stack((-0.5 * score, 0.5 * score), dim=-1)

    def base_logits(self, z):
        return self[1](self[0](z))


class DynamicLinearHead(ResidualHead):
    """s(z) = (w0 + U tanh(V normalize(z)))^T z + b0.

    ``rank`` controls the capacity of the weight generator, NOT a fixed number
    of experts, clusters, or semantic subspaces. Both G5's base head and U/V
    are trainable. Only the DINO backbone is frozen by the experiment runner.
    """

    def __init__(self, linear_head, feature_dim=1024, rank=16):
        super().__init__(linear_head)
        # New parameters must not change the data sampler/augmentation RNG.
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(feature_dim, rank, bias=False)
            self.up = nn.Linear(rank, feature_dim, bias=False)
            nn.init.zeros_(self.up.weight)

    def correction(self, z):
        return self.up(torch.tanh(self.down(F.normalize(z, dim=-1))))

    def effective_weight(self, z):
        return self[1].weight[1] - self[1].weight[0] + self.correction(z)

    def forward(self, z):
        score = (self.correction(z) * z).sum(dim=-1)
        return self.add_score(self.base_logits(z), score)

    def neighborhood_loss(self, z, epsilon):
        """Exact, numerically stable version of the local linearity residual.

        s(z+e)-s(z)-w(z)^T e = [w(z+e)-w(z)]^T (z+e).
        The shared w0/b0 cancel analytically. This avoids subtraction of full
        logits, retains gradients through BOTH generated weights, and needs
        no additional image/backbone pass. All arithmetic must be FP32.
        """
        z = z.float()
        epsilon = epsilon.float()
        neighbor = z + epsilon
        difference = self.correction(neighbor) - self.correction(z)
        residual = (difference * neighbor).sum(dim=-1)
        denominator = epsilon.square().sum(dim=-1) + 1e-8
        return (residual.square() / denominator).mean()


class ResidualMLPHead(ResidualHead):
    """Parameter-matched nonlinear control, not a weight-generating head.

    Width=32 has 32,800 new weights vs. the rank-16 dynamic head's 32,768.
    Both have zero-initialized residual outputs and identical G5 base logits.
    """

    def __init__(self, linear_head, feature_dim=1024, width=32):
        super().__init__(linear_head)
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(feature_dim, width, bias=False)
            self.out = nn.Linear(width, 1, bias=False)
            nn.init.zeros_(self.out.weight)

    def forward(self, z):
        score = self.out(torch.tanh(self.down(F.normalize(z, dim=-1)))).squeeze(-1)
        return self.add_score(self.base_logits(z), score)


def sample_feature_noise(z, relative_radius, generator):
    """Random direction; each epsilon has norm radius * ||z|| (not per axis)."""
    direction = torch.randn(z.shape, device=z.device, dtype=torch.float32,
                            generator=generator)
    return F.normalize(direction, dim=-1) * (
        relative_radius * z.detach().float().norm(dim=-1, keepdim=True)
    )
