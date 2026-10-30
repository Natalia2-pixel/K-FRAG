def v4_source_token(
    full_source_id,
    key,
):

    full_source_id = bytes(
        full_source_id
    )

    digest = hmac.new(
        key,
        V4_SOURCE_DOMAIN
        + b"\x00"
        + full_source_id,
        hashlib.sha256,
    ).digest()

    return digest[
        :V4_TOKEN_BYTES
    ]
