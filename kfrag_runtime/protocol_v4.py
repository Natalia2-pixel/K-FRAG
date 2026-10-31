import hashlib
import hmac

import torch
import torch.nn.functional as F
from reedsolo import RSCodec


# ---------------------------------------------------------------------
# Frozen V4 protocol contract
# ---------------------------------------------------------------------

V4_TOKEN_BYTES = 6
V4_AUTH_BYTES = 6

# RS(16,6): 6 data bytes + 10 parity bytes = 16 symbols.
V4_RS_CODEC = RSCodec(10)


# ---------------------------------------------------------------------
# Byte-bit lookup
# ---------------------------------------------------------------------

BYTE_BITS = torch.tensor(
    [
        [
            (value >> bit) & 1
            for bit in range(7, -1, -1)
        ]
        for value in range(256)
    ],
    dtype=torch.float32,
)


# ---------------------------------------------------------------------
# Byte scoring
# ---------------------------------------------------------------------

@torch.no_grad()
def source_byte_scores_local(final_logits20):
    if final_logits20.ndim != 2:
        raise ValueError(
            f"Expected [N,20], got {tuple(final_logits20.shape)}"
        )

    if final_logits20.shape[1] != 20:
        raise ValueError(
            f"Expected 20 logits, got {final_logits20.shape[1]}"
        )

    source_logits = final_logits20[:, 4:12]

    z = source_logits[:, None, :]

    bits = BYTE_BITS.to(
        device=final_logits20.device,
        dtype=final_logits20.dtype,
    )[None, :, :]

    signed = (
        2.0 * bits - 1.0
    ) * z

    scores = (
        -F.softplus(-signed)
    ).sum(dim=2)

    return scores


@torch.no_grad()
def auth_byte_scores_local(final_logits20):
    if tuple(final_logits20.shape) != (16, 20):
        raise RuntimeError(
            "Expected final_logits20=(16,20), "
            f"got {tuple(final_logits20.shape)}"
        )

    # source_byte_scores_local expects byte bits in positions [4:12].
    # Move AUTH logits [12:20] into those positions.
    proxy = final_logits20.clone()
    proxy[:, 4:12] = final_logits20[:, 12:20]

    scores = source_byte_scores_local(proxy)

    if tuple(scores.shape) != (16, 256):
        raise RuntimeError(
            "Expected AUTH byte scores=(16,256), "
            f"got {tuple(scores.shape)}"
        )

    return scores


# ---------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------

def v4_compute_auth_tag(
    token48,
    key,
    auth_domain,
):
    token48 = bytes(token48)

    if len(token48) != V4_TOKEN_BYTES:
        raise ValueError(
            f"Expected {V4_TOKEN_BYTES}-byte V4 token."
        )

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("key must be bytes.")

    if not isinstance(auth_domain, (bytes, bytearray)):
        raise TypeError("auth_domain must be bytes.")

    digest = hmac.new(
        bytes(key),
        bytes(auth_domain)
        + b"\x00"
        + token48,
        hashlib.sha256,
    ).digest()

    return digest[:V4_AUTH_BYTES]


# ---------------------------------------------------------------------
# Reed-Solomon recovery
# ---------------------------------------------------------------------

def v4_rs_decode(
    codeword,
    erasures=None,
):
    if erasures is None:
        erasures = []

    try:
        result = V4_RS_CODEC.decode(
            bytes(
                int(x) & 0xFF
                for x in codeword
            ),
            erase_pos=[
                int(x)
                for x in erasures
            ],
        )

    except Exception:
        return None

    if isinstance(result, tuple):
        message = bytes(result[0])
    else:
        message = bytes(result)

    if len(message) < V4_TOKEN_BYTES:
        return None

    return message[:V4_TOKEN_BYTES]


def v4_try_rs_configuration(
    observed_bytes,
    known_positions,
):
    known_set = {
        int(p)
        for p in known_positions
    }

    erasures = [
        p
        for p in range(16)
        if p not in known_set
    ]

    return v4_rs_decode(
        observed_bytes,
        erasures,
    )