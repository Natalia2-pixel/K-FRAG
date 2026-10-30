@torch.no_grad()
def recover_v410_at_placement(
    crop,
    candidate_x,
    candidate_y,
):

    candidate_x = int(candidate_x)
    candidate_y = int(candidate_y)

    # ----------------------------------------------------------------------------------------------
    # Reconstruct canonical canvas
    # ----------------------------------------------------------------------------------------------

    canvas = place_crop_on_canvas(
        crop=crop,
        candidate_x=candidate_x,
        candidate_y=candidate_y,
        canonical_size=256,
        neutral_value=0.5,
    )

    regions = split_batch_16_regions_v2(
        canvas
    ).to(
        DEVICE
    )

    visibility = region_visibility_weights(
        crop_height=int(crop.shape[-2]),
        crop_width=int(crop.shape[-1]),
        candidate_x=candidate_x,
        candidate_y=candidate_y,
        device=DEVICE,
    )

    if not torch.is_tensor(visibility):

        visibility = torch.as_tensor(
            visibility,
            dtype=torch.float32,
            device=DEVICE,
        )

    visibility = (
        visibility
        .reshape(16)
        .float()
        .to(DEVICE)
    )


    # ----------------------------------------------------------------------------------------------
    # Exact same feature path used during symbol-head training
    # ----------------------------------------------------------------------------------------------

    decoder = frozen_fragment_features(
        regions,
        visibility,
    )

    final_logits20 = decoder["final"]
    features = decoder["features"]


    # ----------------------------------------------------------------------------------------------
    # Source / auth byte scores
    # ----------------------------------------------------------------------------------------------

    source_baseline = source_byte_scores_local(
        final_logits20
    )

    auth_baseline = auth_byte_scores_local(
        final_logits20
    )

    source_scores = weak_symbol_head(
        features,
        source_baseline,
    )

    auth_scores = auth_symbol_head(
        features,
        auth_baseline,
    )


    # ----------------------------------------------------------------------------------------------
    # Hard Top-1 byte predictions
    # ----------------------------------------------------------------------------------------------

    source_bytes = (
        source_scores
        .argmax(dim=1)
        .detach()
        .cpu()
        .tolist()
    )

    auth_bytes = (
        auth_scores
        .argmax(dim=1)
        .detach()
        .cpu()
        .tolist()
    )


    # ----------------------------------------------------------------------------------------------
    # Select SIX strongest regions by visibility.
    #
    # RS(16,6) fundamentally needs six known symbols.
    # ----------------------------------------------------------------------------------------------

    ranked_positions = torch.argsort(
        visibility,
        descending=True,
    ).tolist()

    known_positions = [
        p
        for p in ranked_positions
        if float(visibility[p].item()) > 0.0
    ][:6]


    if len(known_positions) < 6:

        return {
            "accepted": False,
            "status": "insufficient_visible_regions",
            "x": candidate_x,
            "y": candidate_y,
            "known_positions": known_positions,
            "visibility": visibility,
        }


    # ----------------------------------------------------------------------------------------------
    # Construct observed RS codewords
    # ----------------------------------------------------------------------------------------------

    source_observed = [
        0
        for _ in range(16)
    ]

    auth_observed = [
        0
        for _ in range(16)
    ]


    for p in known_positions:

        source_observed[p] = int(
            source_bytes[p]
        )

        auth_observed[p] = int(
            auth_bytes[p]
        )


    # ----------------------------------------------------------------------------------------------
    # RS(16,6) reconstruction
    # ----------------------------------------------------------------------------------------------

    recovered_token = v4_try_rs_configuration(
        observed_bytes=source_observed,
        known_positions=known_positions,
    )

    recovered_auth = v4_try_rs_configuration(
        observed_bytes=auth_observed,
        known_positions=known_positions,
    )


    if recovered_token is None:

        return {
            "accepted": False,
            "status": "source_rs_failed",
            "x": candidate_x,
            "y": candidate_y,
            "known_positions": known_positions,
            "visibility": visibility,
        }


    if recovered_auth is None:

        return {
            "accepted": False,
            "status": "auth_rs_failed",
            "x": candidate_x,
            "y": candidate_y,
            "known_positions": known_positions,
            "visibility": visibility,
        }


    recovered_token = bytes(
        recovered_token
    )

    recovered_auth = bytes(
        recovered_auth
    )


    # ----------------------------------------------------------------------------------------------
    # Exact cryptographic authentication
    # ----------------------------------------------------------------------------------------------

    expected_auth = v4_compute_auth_tag(
        recovered_token,
        V3H_TRAIN_KEY,
    )

    authenticated = hmac.compare_digest(
        recovered_auth,
        expected_auth,
    )


    # ----------------------------------------------------------------------------------------------
    # Confidence score — only for ranking authenticated placements
    # ----------------------------------------------------------------------------------------------

    source_conf = torch.softmax(
        source_scores,
        dim=1,
    ).max(
        dim=1
    ).values

    auth_conf = torch.softmax(
        auth_scores,
        dim=1,
    ).max(
        dim=1
    ).values


    support = 0.0

    for p in known_positions:

        support += (
            float(
                visibility[p].item()
            )
            *
            0.5
            *
            (
                float(
                    source_conf[p].item()
                )
                +
                float(
                    auth_conf[p].item()
                )
            )
        )


    return {
        "accepted":
            bool(authenticated),

        "status":
            (
                "accepted"
                if authenticated
                else "auth_mismatch"
            ),

        "token":
            (
                recovered_token
                if authenticated
                else None
            ),

        "auth":
            (
                recovered_auth
                if authenticated
                else None
            ),

        "x":
            candidate_x,

        "y":
            candidate_y,

        "known_positions":
            tuple(
                known_positions
            ),

        "support":
            float(
                support
            ),

        "visibility":
            visibility,

        "source_bytes":
            source_bytes,

        "auth_bytes":
            auth_bytes,
    }
