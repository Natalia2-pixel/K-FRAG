def normalize_target_bits(
    target_bits20,
):

    if tuple(target_bits20.shape) == (
        4,
        4,
        20,
    ):

        target_bits20 = target_bits20.reshape(
            16,
            20,
        )

    elif tuple(target_bits20.shape) == (
        1,
        4,
        4,
        20,
    ):

        target_bits20 = target_bits20.reshape(
            16,
            20,
        )

    elif tuple(target_bits20.shape) == (
        1,
        16,
        20,
    ):

        target_bits20 = target_bits20[0]

    elif tuple(target_bits20.shape) == (
        16,
        20,
    ):

        pass

    else:

        raise RuntimeError(
            "Unexpected target_bits20 shape: "
            + str(
                tuple(
                    target_bits20.shape
                )
            )
        )


    return (
        target_bits20
        .to(DEVICE)
        .float()
    )
