def fragments_to_bits20_v49(
    fragments,
):

    if len(fragments) != 16:
        raise ValueError(
            f"Expected 16 fragments, got {len(fragments)}"
        )

    rows = []

    for item in fragments:

        if len(item) != 3:
            raise ValueError(
                "Each fragment must be (index, source_byte, auth_byte)."
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
            +
            source_bits
            +
            auth_bits
        )


    bits20 = torch.tensor(
        rows,
        dtype=torch.float32,
        device=DEVICE,
    )


    if tuple(bits20.shape) != (16, 20):
        raise RuntimeError(
            f"Expected bits20=(16,20), got {tuple(bits20.shape)}"
        )


    return bits20
