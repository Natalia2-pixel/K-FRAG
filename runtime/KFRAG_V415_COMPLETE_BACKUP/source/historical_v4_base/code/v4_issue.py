def v4_issue(
    full_source_id,
    key,
):

    token48 = v4_source_token(
        full_source_id,
        key,
    )

    auth48 = v4_compute_auth_tag(
        token48,
        key,
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
            f"V4 source codeword length = {len(source_symbols)}, expected 16"
        )

    if len(auth_symbols) != 16:
        raise RuntimeError(
            f"V4 auth codeword length = {len(auth_symbols)}, expected 16"
        )


    return [
        (
            i,
            int(source_symbols[i]),
            int(auth_symbols[i]),
        )
        for i in range(16)
    ]
