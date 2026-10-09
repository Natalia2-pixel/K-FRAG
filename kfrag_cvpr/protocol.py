"""K-FRAG V3 authenticated regional-evidence protocol.

The cryptographic construction mirrors the recovered final notebook protocol.
No secret key is included in this repository.
"""

import hashlib
import hmac
from reedsolo import RSCodec

V3_VERSION = 3
V3_N = 16
V3_K = 8
V3_SOURCE_BYTES = 8
V3_AUTH_BYTES = 8
V3_HMAC_DOMAIN = b"KFRAG-V3-SOURCE-GLOBAL"
STRICT_VERIFY_COUNT = 8

V3_RS_CODEC = RSCodec(V3_N - V3_K)
V3_AUTH_RS_CODEC = RSCodec(V3_N - V3_K)


def v3_encode_source_id(source_id):
    if isinstance(source_id, bytearray):
        source_id = bytes(source_id)
    if not isinstance(source_id, bytes):
        raise TypeError("source_id must be bytes")
    if len(source_id) != V3_SOURCE_BYTES:
        raise ValueError(
            f"V3 source_id must be exactly {V3_SOURCE_BYTES} bytes; got {len(source_id)}"
        )
    encoded = bytes(V3_RS_CODEC.encode(source_id))
    if len(encoded) != V3_N:
        raise RuntimeError(f"Expected {V3_N} RS source symbols, got {len(encoded)}")
    return encoded


def v3_compute_auth_tag(source_id, key, source_symbols):
    if isinstance(source_id, bytearray):
        source_id = bytes(source_id)
    if not isinstance(source_id, bytes):
        raise TypeError("source_id must be bytes")
    if len(source_id) != V3_SOURCE_BYTES:
        raise ValueError(f"V3 source_id must be exactly {V3_SOURCE_BYTES} bytes")
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not key:
        raise ValueError("key must be non-empty")
    if isinstance(source_symbols, bytearray):
        source_symbols = bytes(source_symbols)
    elif not isinstance(source_symbols, bytes):
        source_symbols = bytes(source_symbols)
    if len(source_symbols) != V3_N:
        raise ValueError(f"Expected {V3_N} source symbols; got {len(source_symbols)}")
    message = (
        V3_HMAC_DOMAIN
        + bytes([V3_VERSION])
        + len(source_id).to_bytes(2, "big")
        + source_id
        + bytes(range(V3_N))
        + source_symbols
    )
    return hmac.new(key, message, hashlib.sha256).digest()[:V3_AUTH_BYTES]


def v3_encode_auth_tag(auth_tag):
    if isinstance(auth_tag, bytearray):
        auth_tag = bytes(auth_tag)
    if not isinstance(auth_tag, bytes):
        raise TypeError("auth_tag must be bytes")
    if len(auth_tag) != V3_AUTH_BYTES:
        raise ValueError(
            f"V3 auth_tag must be exactly {V3_AUTH_BYTES} bytes; got {len(auth_tag)}"
        )
    encoded = bytes(V3_AUTH_RS_CODEC.encode(auth_tag))
    if len(encoded) != V3_N:
        raise RuntimeError(f"Expected {V3_N} auth symbols, got {len(encoded)}")
    return encoded


def v3_auth_shares(source_id, key, source_symbols=None):
    if source_symbols is None:
        source_symbols = v3_encode_source_id(source_id)
    auth_tag = v3_compute_auth_tag(source_id, key, source_symbols)
    auth_shares = v3_encode_auth_tag(auth_tag)
    if len(auth_shares) != V3_N:
        raise RuntimeError(f"Expected {V3_N} auth shares, got {len(auth_shares)}")
    return auth_shares


def v3_issue(source_id, key):
    source_symbols = v3_encode_source_id(source_id)
    auth_shares = v3_auth_shares(source_id, key, source_symbols)
    return [
        (index, source_symbols[index], auth_shares[index])
        for index in range(V3_N)
    ]


def integer_to_bits_msb(value, width):
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value {value} does not fit in {width} bits")
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def fragments_to_bits20(fragments):
    """Convert the protocol tuples (index, source_symbol, auth_symbol) to 20 bits.

    This is the direct 4+8+8 representation specified by the final protocol.
    """
    if len(fragments) != V3_N:
        raise ValueError(f"Expected {V3_N} fragments, got {len(fragments)}")
    return [
        integer_to_bits_msb(index, 4)
        + integer_to_bits_msb(source_symbol, 8)
        + integer_to_bits_msb(auth_symbol, 8)
        for index, source_symbol, auth_symbol in fragments
    ]
