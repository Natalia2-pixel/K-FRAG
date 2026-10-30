# K-FRAG V4 — RESTORE SYMBOL HEADS
#
# Run AFTER rebuilding the required model classes and decoder architecture.

import torch
from pathlib import Path

BACKUP_ROOT = Path("KFRAG_V4_COMPLETE_BACKUP")

checkpoint = torch.load(
    BACKUP_ROOT / "state" / "v4_symbol_heads.pt",
    map_location=DEVICE,
)

weak_symbol_head.load_state_dict(
    checkpoint["weak_symbol_head_state"]
)

auth_symbol_head.load_state_dict(
    checkpoint["auth_symbol_head_state"]
)

weak_symbol_head.to(DEVICE).eval()
auth_symbol_head.to(DEVICE).eval()

for p in weak_symbol_head.parameters():
    p.requires_grad = False

for p in auth_symbol_head.parameters():
    p.requires_grad = False

print("K-FRAG V4 source/auth heads restored and frozen.")
