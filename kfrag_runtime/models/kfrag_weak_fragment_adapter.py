import torch
import torch.nn as nn
class KFRAGWeakFragmentAdapter(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(61, 128),
            nn.GELU(),

            nn.Linear(128, 96),
            nn.GELU(),

            nn.Linear(96, 64),
            nn.GELU(),

            nn.Linear(64, 20),
        )

        # Start exactly at baseline.
        nn.init.zeros_(
            self.net[-1].weight
        )

        nn.init.zeros_(
            self.net[-1].bias
        )


    def forward(
        self,
        final_logits,
        base_fused,
        raw_mean,
        visibility,
    ):

        if visibility.ndim == 1:
            visibility = visibility[:, None]

        x = torch.cat(
            [
                final_logits,
                base_fused,
                raw_mean,
                visibility,
            ],
            dim=1,
        )

        residual = self.net(
            x
        )

        return (
            final_logits
            +
            residual
        )
