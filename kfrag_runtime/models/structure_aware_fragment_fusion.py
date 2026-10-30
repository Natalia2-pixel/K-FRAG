import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 4. STRUCTURE-AWARE FUSION MODEL
#
# Treat the 4×4 logits as a spatial feature map rather than
# flattening/averaging them immediately.
#
# Also learns one reliability score for every spatial cell.
# ============================================================

class SpatialResidualBlock(
    nn.Module
):

    def __init__(
        self,
        channels,
    ):

        super().__init__()


        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )


        self.norm1 = nn.GroupNorm(
            8,
            channels,
        )


        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )


        self.norm2 = nn.GroupNorm(
            8,
            channels,
        )


    def forward(
        self,
        x,
    ):

        residual = x

        x = F.silu(
            self.norm1(
                self.conv1(
                    x
                )
            )
        )


        x = self.norm2(
            self.conv2(
                x
            )
        )


        return F.silu(
            x
            +
            residual
        )


class StructureAwareFragmentFusion(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        # Input = 20 pretrained bit logits per cell.
        self.input_projection = nn.Conv2d(
            20,
            128,
            kernel_size=1,
        )


        self.spatial = nn.Sequential(

            SpatialResidualBlock(
                128
            ),

            SpatialResidualBlock(
                128
            ),

            SpatialResidualBlock(
                128
            ),

            SpatialResidualBlock(
                128
            ),
        )


        # Reliability score per cell.
        self.reliability = nn.Sequential(

            nn.Conv2d(
                128,
                64,
                kernel_size=1,
            ),

            nn.SiLU(),

            nn.Conv2d(
                64,
                1,
                kernel_size=1,
            ),
        )


        # Global context from all 16 cells.
        self.context = nn.Sequential(

            nn.Linear(
                128,
                256,
            ),

            nn.LayerNorm(
                256
            ),

            nn.SiLU(),

            nn.Linear(
                256,
                256,
            ),

            nn.LayerNorm(
                256
            ),

            nn.SiLU(),
        )


        # Separate semantic heads.
        self.index_head = nn.Linear(
            256,
            4,
        )


        self.source_head = nn.Linear(
            256,
            8,
        )


        self.auth_head = nn.Linear(
            256,
            8,
        )


    def forward(
        self,
        logits20,
    ):

        if tuple(
            logits20.shape[
                1:
            ]
        ) != (
            4,
            4,
            20,
        ):

            raise ValueError(
                "Expected [B,4,4,20]"
            )


        # B,4,4,20 -> B,20,4,4
        x = logits20.permute(
            0,
            3,
            1,
            2,
        )


        x = self.input_projection(
            x
        )


        x = self.spatial(
            x
        )


        # ----------------------------------------------------
        # Learn cell reliability.
        # ----------------------------------------------------

        reliability_logits = self.reliability(
            x
        ).flatten(
            2
        )


        weights = torch.softmax(
            reliability_logits,
            dim=-1,
        )


        # B,C,16
        spatial_features = x.flatten(
            2
        )


        # Weighted combination of 16 cells.
        fused = (
            spatial_features
            *
            weights
        ).sum(
            dim=-1
        )


        fused = self.context(
            fused
        )


        index_logits = self.index_head(
            fused
        )


        source_logits = self.source_head(
            fused
        )


        auth_logits = self.auth_head(
            fused
        )


        logits = torch.cat(
            [
                index_logits,
                source_logits,
                auth_logits,
            ],
            dim=1,
        )


        return {
            "logits20":
                logits,

            "index_logits":
                index_logits,

            "source_logits":
                source_logits,

            "auth_logits":
                auth_logits,

            "cell_weights":
                weights.reshape(
                    -1,
                    4,
                    4,
                ),
        }