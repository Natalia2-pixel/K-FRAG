def split_16_regions(
    image,
):

    if tuple(
        image.shape[
            1:
        ]
    ) != (
        3,
        256,
        256,
    ):
        raise ValueError(
            "Expected [B,3,256,256]"
        )


    tiles = []


    for row in range(
        4
    ):

        for col in range(
            4
        ):

            y0 = row * 64
            x0 = col * 64

            tiles.append(
                image[
                    :,
                    :,
                    y0:y0 + 64,
                    x0:x0 + 64,
                ]
            )


    return torch.cat(
        tiles,
        dim=0,
    )
