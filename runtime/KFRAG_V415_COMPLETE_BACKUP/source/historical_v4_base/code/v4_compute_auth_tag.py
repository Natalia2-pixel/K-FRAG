def v4_compute_auth_tag(
    token48,
    key,
):

    token48 = bytes(
        token48
    )

    if len(token48) != V4_TOKEN_BYTES:
        raise ValueError(
            f"Expected {V4_TOKEN_BYTES}-byte V4 token."
        )

    digest = hmac.new(
        key,
        V4_AUTH_DOMAIN
        + b"\x00"
        + token48,
        hashlib.sha256,
    ).digest()

    return digest[
        :V4_AUTH_BYTES
    ]
