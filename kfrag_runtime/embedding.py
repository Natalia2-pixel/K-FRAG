import hashlib
import hmac

import torch

from .protocol_v4 import (
    V4_RS_CODEC,
    V4_AUTH_DOMAIN,
    V4_SOURCE_DOMAIN,
    V4_TOKEN_BYTES,
    V4_AUTH_BYTES,
    v4_compute_auth_tag,
)

from .runtime_ops import (
    split_batch_16_regions_v2,
)


# ---------------------------------------------------------------------
# Historical embedding defaults
#
# StageDTagCapacityV1.forward defaults recovered from the frozen model:
#
#     amplitude=0.014
#     active_tag_bits=32
#
# No preserved override has been found.
# ---------------------------------------------------------------------

DEFAULT_EMBED_AMPLITUDE = 0.014
DEFAULT_ACTIVE_TAG_BITS = 8


# ---------------------------------------------------------------------
# V4 48-bit source token
# ---------------------------------------------------------------------

def v4_source_token(
    full_source_id,
    key,
):
    """
    Generate the historical keyed 48-bit V4 provenance token.
    """

    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(
            "key must be bytes or bytearray."
        )

    full_source_id = bytes(full_source_id)

    digest = hmac.new(
        bytes(key),
        V4_SOURCE_DOMAIN
        + b"\x00"
        + full_source_id,
        hashlib.sha256,
    ).digest()

    return digest[:V4_TOKEN_BYTES]


# ---------------------------------------------------------------------
# V4 fragment issuance
# ---------------------------------------------------------------------

def v4_issue(
    full_source_id,
    key,
):
    """
    Issue the 16 authenticated V4 fragments.

    Each fragment is:

        (
            logical_index,
            source_RS_symbol,
            authentication_RS_symbol,
        )

    corresponding to:

        4 index bits
        8 source bits
        8 authentication bits
    """

    token48 = v4_source_token(
        full_source_id,
        key,
    )

    auth48 = v4_compute_auth_tag(
        token48,
        key,
        V4_AUTH_DOMAIN,
    )

    source_symbols = bytes(
        V4_RS_CODEC.encode(
            token48
        )
    )

    auth_symbols = bytes(
        V4_RS_CODEC.encode(
            auth48
        )
    )

    if len(source_symbols) != 16:
        raise RuntimeError(
            "V4 source codeword length "
            f"= {len(source_symbols)}, expected 16"
        )

    if len(auth_symbols) != 16:
        raise RuntimeError(
            "V4 auth codeword length "
            f"= {len(auth_symbols)}, expected 16"
        )

    return [
        (
            i,
            int(source_symbols[i]),
            int(auth_symbols[i]),
        )
        for i in range(16)
    ]


# ---------------------------------------------------------------------
# V4 fragments -> exact 20-bit fragment tensors
# ---------------------------------------------------------------------

def fragments_to_bits20_v49(
    fragments,
    device=None,
):
    """
    Convert the 16 V4 fragments to [16,20] binary tensors.

    Bit order is exactly:

        index:  MSB -> LSB, 4 bits
        source: MSB -> LSB, 8 bits
        auth:   MSB -> LSB, 8 bits
    """

    if len(fragments) != 16:
        raise ValueError(
            f"Expected 16 fragments, got {len(fragments)}"
        )

    rows = []

    for item in fragments:

        if len(item) != 3:
            raise ValueError(
                "Each fragment must be "
                "(index, source_byte, auth_byte)."
            )

        index_value = int(item[0])
        source_value = int(item[1])
        auth_value = int(item[2])

        if not (0 <= index_value < 16):
            raise ValueError(
                f"Invalid fragment index: {index_value}"
            )

        if not (0 <= source_value <= 255):
            raise ValueError(
                f"Invalid source byte: {source_value}"
            )

        if not (0 <= auth_value <= 255):
            raise ValueError(
                f"Invalid auth byte: {auth_value}"
            )

        index_bits = [
            (index_value >> shift) & 1
            for shift in [3, 2, 1, 0]
        ]

        source_bits = [
            (source_value >> shift) & 1
            for shift in range(7, -1, -1)
        ]

        auth_bits = [
            (auth_value >> shift) & 1
            for shift in range(7, -1, -1)
        ]

        rows.append(
            index_bits
            + source_bits
            + auth_bits
        )

    bits20 = torch.tensor(
        rows,
        dtype=torch.float32,
        device=device,
    )

    if tuple(bits20.shape) != (16, 20):
        raise RuntimeError(
            "Expected bits20=(16,20), "
            f"got {tuple(bits20.shape)}"
        )

    return bits20


# ---------------------------------------------------------------------
# 16 x 64x64 tiles -> one 256x256 image
# ---------------------------------------------------------------------

def assemble_16_regions(
    tiles,
):
    """
    Exact inverse of the historical 4x4 image tiling.
    """

    if tuple(tiles.shape[1:]) != (
        3,
        64,
        64,
    ):
        raise ValueError(
            "Expected [16,3,64,64]"
        )

    if len(tiles) != 16:
        raise ValueError(
            f"Expected 16 tiles, got {len(tiles)}"
        )

    rows = []

    for row in range(4):

        row_tiles = [
            tiles[
                row * 4 + col:
                row * 4 + col + 1
            ]
            for col in range(4)
        ]

        rows.append(
            torch.cat(
                row_tiles,
                dim=3,
            )
        )

    return torch.cat(
        rows,
        dim=2,
    )


# ---------------------------------------------------------------------
# Full V4 embedding
# ---------------------------------------------------------------------

@torch.no_grad()
def embed_full_spatial_image_v4(
    clean_image,
    source_id,
    runtime,
    auth_key,
    amplitude=DEFAULT_EMBED_AMPLITUDE,
    active_tag_bits=DEFAULT_ACTIVE_TAG_BITS,
):
    """
    Embed one authenticated K-FRAG V4 payload into a 256x256 image.

    Parameters
    ----------
    clean_image:
        [1,3,256,256] or [3,256,256].

    source_id:
        Original source identifier as bytes.

    runtime:
        Runtime returned by load_v415_runtime(). It must contain the
        independently restored trained ``v2_model_audit``.

    auth_key:
        Private 32-byte HMAC key. It is supplied externally and is never
        stored by this module.

    amplitude:
        V2 watermark amplitude. Historical StageDTagCapacityV1 default
        is 0.014.

    active_tag_bits:
        Historical StageDTagCapacityV1 default is 32.
    """

    if not isinstance(runtime, dict):
        raise TypeError(
            "runtime must be a dictionary."
        )

    if "v2_model_audit" not in runtime:
        raise KeyError(
            "runtime does not contain v2_model_audit."
        )

    if "device" not in runtime:
        raise KeyError(
            "runtime does not contain device."
        )

    if not isinstance(auth_key, (bytes, bytearray)):
        raise TypeError(
            "auth_key must be bytes or bytearray."
        )

    if len(auth_key) != 32:
        raise ValueError(
            "K-FRAG authentication key must contain 32 bytes."
        )

    device = runtime["device"]
    v2_model_audit = runtime["v2_model_audit"]

    # ---------------------------------------------------------
    # Normalize image
    # ---------------------------------------------------------

    if clean_image.ndim == 3:
        clean_image = clean_image.unsqueeze(0)

    if tuple(clean_image.shape) != (
        1,
        3,
        256,
        256,
    ):
        raise ValueError(
            "clean_image must have shape "
            "[3,256,256] or [1,3,256,256], "
            f"got {tuple(clean_image.shape)}"
        )

    clean_image = clean_image.to(
        device=device,
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # V4 packet issuance
    # ---------------------------------------------------------

    fragments = v4_issue(
        source_id,
        bytes(auth_key),
    )

    bits20 = fragments_to_bits20_v49(
        fragments,
        device=device,
    )

    # ---------------------------------------------------------
    # Existing 4x4 image tiling
    # ---------------------------------------------------------

    tiles = split_batch_16_regions_v2(
        clean_image
    ).to(device)

    if tuple(tiles.shape) != (
        16,
        3,
        64,
        64,
    ):
        raise RuntimeError(
            "Expected 16 image tiles, got "
            f"{tuple(tiles.shape)}"
        )

    # ---------------------------------------------------------
    # Repeat each 20-bit regional payload across its local
    # 4x4 packet cells.
    # ---------------------------------------------------------

    payload20 = (
        bits20[
            :,
            None,
            None,
            :
        ]
        .expand(
            -1,
            4,
            4,
            -1,
        )
        .contiguous()
    )

    # ---------------------------------------------------------
    # Preserve historical 44-bit V2 network interface.
    #
    # V4 uses the first 20 bits.
    # Remaining positions stay zero.
    # ---------------------------------------------------------

    packet44 = torch.zeros(
        16,
        4,
        4,
        44,
        device=device,
        dtype=payload20.dtype,
    )

    packet44[
        ...,
        :20
    ] = payload20

    # ---------------------------------------------------------
    # Historical trained V2 watermark encoder
    # ---------------------------------------------------------

    output = v2_model_audit(
        tiles,
        packet44,
        amplitude=float(amplitude),
        active_tag_bits=int(active_tag_bits),
    )

    if not isinstance(output, dict):
        raise RuntimeError(
            "V2 encoder did not return a dictionary."
        )

    if "watermarked_image" not in output:
        raise RuntimeError(
            "V2 encoder output is missing "
            "'watermarked_image'."
        )

    watermarked_tiles = output[
        "watermarked_image"
    ]

    if tuple(watermarked_tiles.shape) != (
        16,
        3,
        64,
        64,
    ):
        raise RuntimeError(
            "Expected watermarked tiles "
            "(16,3,64,64), got "
            f"{tuple(watermarked_tiles.shape)}"
        )

    full_watermarked = assemble_16_regions(
        watermarked_tiles
    )

    # ---------------------------------------------------------
    # Independent protocol consistency check
    # ---------------------------------------------------------

    expected_fragments = v4_issue(
        source_id,
        bytes(auth_key),
    )

    expected_bits = fragments_to_bits20_v49(
        expected_fragments,
        device=device,
    )

    if not torch.equal(
        bits20.to(torch.int64),
        expected_bits.to(torch.int64),
    ):
        raise RuntimeError(
            "V4 embedding consistency failed: "
            "embedded target does not equal issued packet."
        )

    return {
        "watermarked": full_watermarked,
        "watermarked_tiles": watermarked_tiles,
        "target_bits20": bits20,
        "fragments": fragments,
        "token": v4_source_token(
            source_id,
            bytes(auth_key),
        ),
        "auth": v4_compute_auth_tag(
            v4_source_token(
                source_id,
                bytes(auth_key),
            ),
            bytes(auth_key),
            V4_AUTH_DOMAIN,
        ),
        "version": "V4",
        "amplitude": float(amplitude),
        "active_tag_bits": int(active_tag_bits),
    }


__all__ = [
    "DEFAULT_EMBED_AMPLITUDE",
    "DEFAULT_ACTIVE_TAG_BITS",
    "v4_source_token",
    "v4_issue",
    "fragments_to_bits20_v49",
    "assemble_16_regions",
    "embed_full_spatial_image_v4",
]