def v4_try_rs_configuration(
    observed_bytes,
    known_positions,
):

    known_set = set(
        int(p)
        for p in known_positions
    )

    erasures = [
        p
        for p in range(16)
        if p not in known_set
    ]

    recovered = v4_rs_decode(
        observed_bytes,
        erasures,
    )

    return recovered
