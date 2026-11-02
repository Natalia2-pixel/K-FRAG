from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from kfrag_runtime.runtime_loader import load_v415_runtime
from kfrag_runtime.embedding import embed_full_spatial_image_v4
from kfrag_runtime.verify import verify_crop_v415
from kfrag_runtime.protocol_v4 import V4_AUTH_DOMAIN


# ------------------------------------------------------------
# Load runtime
# ------------------------------------------------------------

runtime = load_v415_runtime(
    ROOT / "checkpoints" / "KFRAG_V415_FINAL_MODELS.pt"
)

print("runtime:", runtime["system"])
print("device:", runtime["device"])


# ------------------------------------------------------------
# Deterministic PUBLIC test key
#
# This is intentionally not the private research key.
# ------------------------------------------------------------

test_key = bytes(range(32))

source_id = b"KFRAG-PUBLIC-END-TO-END-TEST"


# ------------------------------------------------------------
# Deterministic smooth synthetic carrier
#
# Avoid white-noise images: K-FRAG was trained/evaluated on
# natural-image statistics, so a low-frequency carrier is a
# better public regression input.
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
# Embed
# ------------------------------------------------------------

embedded = embed_full_spatial_image_v4(
    clean_image=clean,
    source_id=source_id,
    runtime=runtime,
    auth_key=test_key,
)

watermarked = embedded["watermarked"]

print(
    "watermarked:",
    tuple(watermarked.shape),
)


# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

result = verify_crop_v415(
    crop=watermarked[0],
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
# Token consistency
# ------------------------------------------------------------

if result.get("accepted") is not True:
    raise AssertionError(
        "Public round-trip authentication failed: "
        f"status={result.get('status')!r}"
    )

recovery = result.get("recovery")

if not isinstance(recovery, dict):
    raise AssertionError(
        "Accepted verification did not contain recovery information."
    )
recovered_token = recovery.get("token")

if recovered_token is None:
    raise AssertionError(
        "Verification did not recover a token."
    )

token_match = (
    bytes(recovered_token)
    == bytes(embedded["token"])
)

print("token matches:", token_match)


# ------------------------------------------------------------
# Assertions
# ------------------------------------------------------------

assert result.get("accepted") is True
assert result.get("status") == "accepted"
assert token_match is True

assert result.get("x") == 0
assert result.get("y") == 0

print("PUBLIC END-TO-END ROUNDTRIP TEST: PASS")