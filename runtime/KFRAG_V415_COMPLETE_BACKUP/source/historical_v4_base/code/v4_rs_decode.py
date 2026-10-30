def v4_rs_decode(
    codeword,
    erasures=None,
):

    if erasures is None:
        erasures = []

    try:

        result = V4_RS_CODEC.decode(
            bytes(
                int(x) & 0xFF
                for x in codeword
            ),
            erase_pos=[
                int(x)
                for x in erasures
            ],
        )

    except Exception:

        return None


    if isinstance(
        result,
        tuple,
    ):

        message = bytes(
            result[0]
        )

    else:

        message = bytes(
            result
        )


    if len(message) < 6:
        return None


    return message[:6]
