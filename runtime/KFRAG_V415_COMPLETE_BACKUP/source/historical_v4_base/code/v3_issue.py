def v3_issue(
    source_id,
    key,
):

    source_symbols = (
        v3_encode_source_id(
            source_id
        )
    )


    auth_shares = (
        v3_auth_shares(
            source_id,
            key,
        )
    )


    return [
        (
            index,
            source_symbols[
                index
            ],
            auth_shares[
                index
            ],
        )
        for index
        in range(
            16
        )
    ]
