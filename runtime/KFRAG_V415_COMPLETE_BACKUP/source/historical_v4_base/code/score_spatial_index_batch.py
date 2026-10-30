def score_spatial_index_batch(
    crop,
    placements,
):

    if len(
        placements
    ) == 0:

        return []


    crop3 = (
        crop[0]
        if crop.ndim == 4
        else crop
    )


    _, crop_h, crop_w = (
        crop3.shape
    )


    canvases = _batch_candidate_canvases(
        crop=crop3,
        placements=placements,
    )


    tiles = split_batch_16_regions_v2(
        canvases
    )


    decoded = decode_phase_tiles_v2(
        tiles
    )


    # --------------------------------------------------------------------------------------------
    # FINAL 100K DECODER CONTRACT COMPATIBILITY
    # --------------------------------------------------------------------------------------------

    if isinstance(
        decoded,
        dict,
    ):

        logits20 = decoded[
            "final_logits20"
        ]

    elif torch.is_tensor(
        decoded
    ):

        logits20 = decoded

    else:

        raise TypeError(
            "decode_phase_tiles_v2 returned unsupported type: "
            f"{type(decoded).__name__}"
        )


    B = len(
        placements
    )


    logits20 = logits20.reshape(
        B,
        16,
        20,
    )


    index_logits = logits20[
        :,
        :,
        0:4,
    ]


    index_probs = torch.sigmoid(
        index_logits
    )


    target_bits = (
        canonical_index_bits(
            device=
                index_probs.device,

            dtype=
                index_probs.dtype,
        )
        .unsqueeze(
            0
        )
    )


    # Probability assigned to correct index bit
    bit_match_prob = (
        index_probs
        *
        target_bits
        +
        (
            1.0
            -
            index_probs
        )
        *
        (
            1.0
            -
            target_bits
        )
    )


    # Region-level probabilistic index agreement
    region_prob_score = (
        bit_match_prob.mean(
            dim=-1
        )
    )


    # Hard exact 4-bit region index match
    hard_bits = (
        index_probs
        >=
        0.5
    ).to(
        target_bits.dtype
    )


    hard_region_match = (
        hard_bits
        ==
        target_bits
    ).all(
        dim=-1
    ).float()


    # Index confidence
    bit_confidence = (
        torch.abs(
            index_probs
            -
            0.5
        )
        *
        2.0
    )


    region_confidence = (
        bit_confidence.mean(
            dim=-1
        )
    )


    results = []


    for i, (
        x,
        y,
    ) in enumerate(
        placements
    ):

        visibility = (
            region_visibility_weights(
                crop_height=
                    crop_h,

                crop_width=
                    crop_w,

                candidate_x=
                    x,

                candidate_y=
                    y,

                device=
                    index_probs.device,
            )
        )


        denom = (
            visibility.sum()
            +
            1e-8
        )


        protocol_score = (
            (
                region_prob_score[
                    i
                ]
                *
                visibility
            ).sum()
            /
            denom
        )


        hard_alignment = (
            (
                hard_region_match[
                    i
                ]
                *
                visibility
            ).sum()
            /
            denom
        )


        confidence = (
            (
                region_confidence[
                    i
                ]
                *
                visibility
            ).sum()
            /
            denom
        )


        results.append(
            {
                "x":
                    int(
                        x
                    ),

                "y":
                    int(
                        y
                    ),

                "score":
                    float(
                        protocol_score.item()
                    ),

                "hard_alignment":
                    float(
                        hard_alignment.item()
                    ),

                "confidence":
                    float(
                        confidence.item()
                    ),
            }
        )


    return results
