"""Residual policy network: sees the ball field (world grid), the balls around the robot (ego grid) and a feature
vector, outputs a Gaussian over the residual u ∈ R^k plus a state value.

The actor mean layer starts at exactly zero, so an untrained network reproduces the ghost tracker (u = 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .obs import EGO_N, FIELD_H, FIELD_W, RES_EGO_C, RES_FIELD_C, RES_VEC_DIM


class ResidualNet(nn.Module):
    def __init__(self, k: int = 2, vec_dim: int = RES_VEC_DIM, log_std0: float = -1.0):
        super().__init__()
        self.k = k
        self.field = nn.Sequential(
            nn.Conv2d(RES_FIELD_C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        self.ego = nn.Sequential(
            nn.Conv2d(RES_EGO_C, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            nf = self.field(torch.zeros(1, RES_FIELD_C, FIELD_H, FIELD_W)).shape[1]
            ne = self.ego(torch.zeros(1, RES_EGO_C, EGO_N, EGO_N)).shape[1]
        self.field_fc = nn.Sequential(nn.Linear(nf, 192), nn.LayerNorm(192), nn.Tanh())
        self.ego_fc = nn.Sequential(nn.Linear(ne, 128), nn.LayerNorm(128), nn.Tanh())
        self.vec_fc = nn.Sequential(nn.Linear(vec_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        trunk = 192 + 128 + 128
        self.pi = nn.Sequential(nn.Linear(trunk, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, k)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        self.log_std = nn.Parameter(torch.full((k,), log_std0))
        self.v = nn.Sequential(nn.Linear(trunk, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def features(self, field, ego, vec):
        return torch.cat([self.field_fc(self.field(field)), self.ego_fc(self.ego(ego)), self.vec_fc(vec)], dim=-1)

    def forward(self, field, ego, vec):
        h = self.features(field, ego, vec)
        mu = self.mu(self.pi(h))
        std = self.log_std.clamp(-3.0, -0.3).exp().expand_as(mu)
        return mu, std, self.v(h).squeeze(-1)

    def dist(self, field, ego, vec):
        mu, std, v = self(field, ego, vec)
        return torch.distributions.Normal(mu, std), v
