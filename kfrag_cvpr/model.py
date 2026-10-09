"""K-FRAG final regional-refiner architecture.

This file is a direct modularization of the architecture used by the final
Crop-Hard V3 trajectory. It does not alter layer definitions or the forward rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleRegionalEncoder(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.branch_local = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.branch_context = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        local = F.adaptive_avg_pool2d(self.branch_local(x), (4, 4))
        dilated = F.adaptive_avg_pool2d(self.branch_dilated(x), (4, 4))
        context = F.adaptive_avg_pool2d(self.branch_context(x), (4, 4))
        fused = torch.cat([local, dilated, context], dim=1)
        return self.fuse(fused)


class KFRAGRegionalRefinerV2(nn.Module):
    def __init__(self, input_bits=20, image_features=64, hidden=128):
        super().__init__()
        self.image_encoder = MultiScaleRegionalEncoder(out_channels=image_features)
        self.cell_projection = nn.Sequential(
            nn.Conv2d(input_bits + image_features, hidden, kernel_size=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
        )
        self.general_residual_head = nn.Conv2d(hidden, 20, kernel_size=1)
        self.auth_refinement = nn.Sequential(
            nn.Conv2d(hidden, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 8, kernel_size=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(hidden, 64, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, region_images, raw_logits20):
        image_features = self.image_encoder(region_images)
        raw_cf = raw_logits20.permute(0, 3, 1, 2)
        x = torch.cat([raw_cf, image_features], dim=1)
        x = self.cell_projection(x)
        residual_features = self.refinement(x)
        x = F.silu(x + residual_features)
        general_delta = self.general_residual_head(x)
        refined_cf = raw_cf + general_delta
        auth_delta = self.auth_refinement(x)
        refined_cf[:, 12:20, :, :] = refined_cf[:, 12:20, :, :] + auth_delta
        confidence_logits = self.confidence_head(x).squeeze(1)
        cell_weights = torch.softmax(
            confidence_logits.reshape(confidence_logits.shape[0], -1), dim=1
        ).reshape(-1, 4, 4)
        refined_cell_logits20 = refined_cf.permute(0, 2, 3, 1)
        fragment_logits20 = (
            refined_cell_logits20 * cell_weights.unsqueeze(-1)
        ).sum(dim=(1, 2))
        return {
            "refined_cell_logits20": refined_cell_logits20,
            "confidence_logits": confidence_logits,
            "cell_weights": cell_weights,
            "fragment_logits20": fragment_logits20,
        }
