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


# ------------------------------------------------------------
# Public deterministic test key
# ------------------------------------------------------------

test_key = bytes(range(32))
source_id = b"KFRAG-PUBLIC-CROP-RECOVERY-TEST"


# ------------------------------------------------------------
# Deterministic smooth carrier
# ------------------------------------------------------------

y = torch.linspace(
    0.0,
    1.0,
    256,
).view(256, 1)

x = torch.linspace(
    0.0,
    1.0,
    256,
).view(1, 256)

red = (
    0.30
    + 0.45 * x
    + 0.10 * y
)

green = (
    0.20
    + 0.55 * y
    + 0.05 * x
)

blue = (
    0.25
    + 0.25 * x
    + 0.25 * y
)

clean = torch.stack(
    [
        red.expand(256, 256),
        green.expand(256, 256),
        blue.expand(256, 256),
    ],
    dim=0,
)

clean = (
    clean
    .clamp(0.0, 1.0)
    .unsqueeze(0)
)


# ------------------------------------------------------------
# Embed full image
# ------------------------------------------------------------

embedded = embed_full_spatial_image_v4(
    clean_image=clean,
    source_id=source_id,
    runtime=runtime,
    auth_key=test_key,
)

watermarked = embedded["watermarked"][0]

assert tuple(watermarked.shape) == (
    3,
    256,
    256,
)

print(
    "watermarked:",
    tuple(watermarked.shape),
)


# ------------------------------------------------------------
# Controlled 192 x 192 crop
#
# This is 75% retained in each spatial dimension and
# 56.25% retained by total image area.
#
# True canonical placement:
#
#     x = 32
#     y = 32
# ------------------------------------------------------------

crop_x = 32
crop_y = 32
crop_size = 192

crop = watermarked[
    :,
    crop_y:crop_y + crop_size,
    crop_x:crop_x + crop_size,
]

assert tuple(crop.shape) == (
    3,
    192,
    192,
)

print(
    "crop:",
    tuple(crop.shape),
)

print(
    "true placement:",
    crop_x,
    crop_y,
)


# ------------------------------------------------------------
# Blind crop verification
# ------------------------------------------------------------

result = verify_crop_v415(
    crop=crop,
    runtime=runtime,
    auth_key=test_key,
    auth_domain=V4_AUTH_DOMAIN,
    top_k=20,
)

print("status:", result.get("status"))
print("accepted:", result.get("accepted"))
print("rank:", result.get("rank"))
print("x:", result.get("x"))
print("y:", result.get("y"))


# ------------------------------------------------------------
# Validate authenticated recovery
# ------------------------------------------------------------

if result.get("accepted") is not True:
    raise AssertionError(
        "Public crop authentication failed: "
        f"status={result.get('status')!r}"
    )

recovery = result.get("recovery")

if not isinstance(recovery, dict):
    raise AssertionError(
        "Accepted crop verification did not contain recovery data."
    )

recovered_token = recovery.get("token")

if recovered_token is None:
    raise AssertionError(
        "Crop verification did not recover a token."
    )

token_match = (
    bytes(recovered_token)
    == bytes(embedded["token"])
)

print(
    "token matches:",
    token_match,
)


print(
    "true placement:",
    crop_x,
    crop_y,
)

print(
    "recovered placement:",
    result.get("x"),
    result.get("y"),
)

dx = abs(result.get("x") - crop_x)
dy = abs(result.get("y") - crop_y)

print(
    "placement error:",
    dx,
    dy,
)

assert result.get("accepted") is True
assert token_match is True

print(
    "PUBLIC BLIND CROP RECOVERY TEST: PASS"
)