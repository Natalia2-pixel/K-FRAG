def v3_fragments_to_bits20(
    fragments,
):

    if len(
        fragments
    ) != 16:
        raise ValueError(
            "Expected 16 fragments"
        )


    bits = []


    for (
        index,
        source_symbol,
        auth_share,
    ) in fragments:

        fragment_bits = (
            integer_to_bits_msb(
                index,
                4,
            )
            +
            integer_to_bits_msb(
                source_symbol,
                8,
            )
            +
            integer_to_bits_msb(
                auth_share,
                8,
            )
        )


        bits.append(
            fragment_bits
        )


    return torch.tensor(
        bits,
        dtype=torch.float32,
    ).reshape(
        4,
        4,
        20,
    )
