def bits8_to_byte(bits8):

    bits8 = (
        bits8
        .to(torch.int64)
    )

    weights = torch.tensor(
        [
            128,
            64,
            32,
            16,
            8,
            4,
            2,
            1,
        ],
        dtype=torch.int64,
        device=bits8.device,
    )

    return (
        bits8
        *
        weights[None, :]
    ).sum(
        dim=1
    )
