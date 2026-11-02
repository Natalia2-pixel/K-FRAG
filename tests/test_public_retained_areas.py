"""
Slow integration evaluation.

Runs full blind synchronization across multiple crop sizes.
Expected to be significantly faster on CUDA than CPU.

This script is intended for manual reproducibility evaluation,
not routine unit-test execution.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from kfrag_runtime.runtime_loader import load_v415_runtime
from kfrag_runtime.embedding import embed_full_spatial_image_v4
from kfrag_runtime.verify import verify_crop_v415
from kfrag_runtime.protocol_v4 import V4_AUTH_DOMAIN


runtime = load_v415_runtime(
    ROOT / "checkpoints" / "KFRAG_V415_FINAL_MODELS.pt"
)

print("runtime:", runtime["system"])
print("device:", runtime["device"])

test_key = bytes(range(32))
source_id = b"KFRAG-PUBLIC-RETAINED-AREA-TEST"


# ------------------------------------------------------------
# Deterministic smooth carrier
# ------------------------------------------------------------

y = torch.linspace(0.0, 1.0, 256).view(256, 1)
x = torch.linspace(0.0, 1.0, 256).view(1, 256)

red = 0.30 + 0.45 * x + 0.10 * y
green = 0.20 + 0.55 * y + 0.05 * x
blue = 0.25 + 0.25 * x + 0.25 * y

clean = torch.stack(
    [
        red.expand(256, 256),
        green.expand(256, 256),
        blue.expand(256, 256),
    ],
    dim=0,
).clamp(0.0, 1.0).unsqueeze(0)


# ------------------------------------------------------------
# Embed once
# ------------------------------------------------------------

embedded = embed_full_spatial_image_v4(
    clean_image=clean,
    source_id=source_id,
    runtime=runtime,
    auth_key=test_key,
)

watermarked = embedded["watermarked"][0]
expected_token = bytes(embedded["token"])

print("watermarked:", tuple(watermarked.shape))


# ------------------------------------------------------------
# Centered crop sizes
# ------------------------------------------------------------

crop_sizes = [
    256,
    240,
    224,
    208,
    192,
    176,
    160,
]

print()
print("=" * 100)
print("PUBLIC RETAINED-AREA EVALUATION")
print("=" * 100)

results = []

for crop_size in crop_sizes:

    crop_x = (256 - crop_size) // 2
    crop_y = (256 - crop_size) // 2

    crop = watermarked[
        :,
        crop_y:crop_y + crop_size,
        crop_x:crop_x + crop_size,
    ]

    retained_linear = crop_size / 256.0
    retained_area = retained_linear * retained_linear

    print()
    print("-" * 100)
    print(
        f"crop={crop_size}x{crop_size} | "
        f"linear={retained_linear * 100:.2f}% | "
        f"area={retained_area * 100:.2f}% | "
        f"true=({crop_x},{crop_y})"
    )

    result = verify_crop_v415(
        crop=crop,
        runtime=runtime,
        auth_key=test_key,
        auth_domain=V4_AUTH_DOMAIN,
        top_k=20,
    )

    accepted = bool(result.get("accepted"))
    status = result.get("status")
    recovered_x = result.get("x")
    recovered_y = result.get("y")

    token_match = False

    recovery = result.get("recovery")

    if isinstance(recovery, dict):
        token = recovery.get("token")

        if token is not None:
            token_match = bytes(token) == expected_token

    if (
        recovered_x is not None
        and recovered_y is not None
    ):
        dx = abs(int(recovered_x) - crop_x)
        dy = abs(int(recovered_y) - crop_y)
    else:
        dx = None
        dy = None

    print("status:", status)
    print("accepted:", accepted)
    print("token matches:", token_match)
    print(
        "recovered placement:",
        recovered_x,
        recovered_y,
    )
    print(
        "placement error:",
        dx,
        dy,
    )

    results.append(
        {
            "crop_size": crop_size,
            "linear_retained": retained_linear,
            "area_retained": retained_area,
            "accepted": accepted,
            "token_match": token_match,
            "x": recovered_x,
            "y": recovered_y,
            "dx": dx,
            "dy": dy,
            "status": status,
        }
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for r in results:
    print(
        f"{r['crop_size']:3d}x{r['crop_size']:3d} | "
        f"area={r['area_retained'] * 100:6.2f}% | "
        f"accepted={str(r['accepted']):5s} | "
        f"token={str(r['token_match']):5s} | "
        f"dx={str(r['dx']):>4s} | "
        f"dy={str(r['dy']):>4s} | "
        f"status={r['status']}"
    )

print()
print("PUBLIC RETAINED-AREA EVALUATION COMPLETE")