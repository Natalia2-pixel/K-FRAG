@torch.no_grad()
def source_byte_scores_local(
    final_logits20,
):

    source_logits = final_logits20[
        :,
        4:12
    ]


    z = source_logits[
        :,
        None,
        :
    ]


    bits = BYTE_BITS[
        None,
        :,
        :
    ]


    signed = (
        2.0 * bits
        -
        1.0
    ) * z


    scores = (
        -F.softplus(
            -signed
        )
    ).sum(
        dim=2
    )


    return scores
