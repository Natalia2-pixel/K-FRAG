import torch.nn as nn


class WeakSymbolResidualHead(nn.Module):
    def __init__(
        self,
        feature_dim=141,
        hidden1=256,
        hidden2=192,
        output_dim=256,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, features, baseline_scores):
        residual = self.body(features)
        return baseline_scores + residual