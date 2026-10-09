"""Final K-FRAG 100K decoder fusion rule.

This module does not instantiate or modify the base decoder/fusion models.
They are supplied by the caller from the existing K-FRAG implementation.
"""

import torch


@torch.no_grad()
def final_logits_100k(
    tiles,
    verification_model,
    extract_raw_logits20,
    old_fusion_logits20,
):
    raw_logits20 = extract_raw_logits20(tiles)
    base_fused = old_fusion_logits20(raw_logits20)
    raw_mean = raw_logits20.mean(dim=(1, 2))
    refined = verification_model(tiles, raw_logits20)
    final_logits20 = (
        base_fused
        + refined["fragment_logits20"]
        - raw_mean
    )
    return {
        "final_logits20": final_logits20,
        "raw_logits20": raw_logits20,
        "fragment_logits20": refined["fragment_logits20"],
        "cell_weights": refined["cell_weights"],
        "confidence_logits": refined["confidence_logits"],
    }
