from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from kfrag_runtime.runtime_loader import load_v415_runtime
from kfrag_runtime.embedding import (
    v4_issue,
    fragments_to_bits20_v49,
    embed_full_spatial_image_v4,
)


runtime = load_v415_runtime(
    r".\checkpoints\KFRAG_V415_FINAL_MODELS.pt"
)

print("runtime:", runtime["system"])
print("device:", runtime["device"])

v2 = runtime["v2_model_audit"]

print("v2 class:", type(v2).__name__)
print(
    "v2 parameters:",
    sum(p.numel() for p in v2.parameters()),
)

# Public deterministic test key.
# This is NOT the private research key.
test_key = bytes(range(32))

fragments = v4_issue(
    b"KFRAG-PUBLIC-REGRESSION-TEST",
    test_key,
)

bits20 = fragments_to_bits20_v49(
    fragments
)

print("fragments:", len(fragments))
print("bits20:", tuple(bits20.shape))

assert len(fragments) == 16
assert tuple(bits20.shape) == (16, 20)

# Synthetic clean image for API/runtime test.
torch.manual_seed(1234)

clean = torch.rand(
    1,
    3,
    256,
    256,
)

embedded = embed_full_spatial_image_v4(
    clean_image=clean,
    source_id=b"KFRAG-PUBLIC-REGRESSION-TEST",
    runtime=runtime,
    auth_key=test_key,
)

assert tuple(embedded["watermarked"].shape) == (
    1,
    3,
    256,
    256,
)

assert len(embedded["fragments"]) == 16
assert tuple(embedded["target_bits20"].shape) == (16, 20)

print("watermarked:", tuple(embedded["watermarked"].shape))
print("PUBLIC REGRESSION TEST: PASS")