def assemble_16_regions(
    tiles,
):

    if tuple(
        tiles.shape[
            1:
        ]
    ) != (
        3,
        64,
        64,
    ):
        raise ValueError(
            "Expected [16,3,64,64]"
        )


    rows = []


    for row in range(
        4
    ):

        row_tiles = [
            tiles[
                row * 4 + col:
                row * 4 + col + 1
            ]
            for col in range(
                4
            )
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
