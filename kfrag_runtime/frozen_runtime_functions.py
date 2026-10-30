
# ====================================================================================================================
# decode_phase_tiles_kfrag_final_100k
# ====================================================================================================================
@torch.no_grad()
def decode_phase_tiles_kfrag_final_100k(tiles):

    tiles = tiles.to(DEVICE)

    raw_logits20 = extract_raw_logits20(
        tiles
    )

    base_fused = old_fusion_logits20(
        raw_logits20
    )

    raw_mean = raw_logits20.mean(
        dim=(1, 2)
    )

    refined = verification_model(
        tiles,
        raw_logits20,
    )

    final_logits20 = (
        base_fused
        + refined["fragment_logits20"]
        - raw_mean
    )

    return final_logits20


# ====================================================================================================================
# frozen_fragment_features
# ====================================================================================================================
@torch.no_grad()
def frozen_fragment_features(
    regions,
    visibility,
):

    regions = regions.to(
        DEVICE
    )


    raw = extract_raw_logits20(
        regions
    )


    raw_mean = raw.mean(
        dim=(1, 2)
    )


    raw_std = raw.std(
        dim=(1, 2),
        unbiased=False,
    )


    raw_max = raw.amax(
        dim=(1, 2)
    )


    raw_min = raw.amin(
        dim=(1, 2)
    )


    base = old_fusion_logits20(
        raw
    )


    refined = verification_model(
        regions,
        raw,
    )[
        "fragment_logits20"
    ]


    final = (
        base
        +
        refined
        -
        raw_mean
    )


    visibility = (
        visibility
        .reshape(
            16,
            1
        )
        .float()
        .to(DEVICE)
    )


    features = torch.cat(
        [
            raw_mean,
            raw_std,
            raw_max,
            raw_min,
            base,
            refined,
            final,
            visibility,
        ],
        dim=1,
    )


    assert features.shape == (
        16,
        141,
    )


    return {
        "features":
            features,

        "raw_mean":
            raw_mean,

        "base":
            base,

        "refined":
            refined,

        "final":
            final,
    }


# ====================================================================================================================
# blind_v412_phase_recovery
# ====================================================================================================================
@torch.no_grad()
def blind_v412_phase_recovery(
    crop,
    top_k=5,
    phase_step=16,
    canonical_size=256,
):

    # ==============================================================================================
    # BLIND SEARCH
    # ==============================================================================================

    raw_candidates = blind_crop_search_spatial_fast(
        crop
    )

    candidates = normalize_blind_candidates(
        raw_candidates
    )

    if len(candidates) == 0:

        return {
            "accepted": False,
            "ambiguous": False,
            "candidate": None,
            "tested_placements": 0,
            "base_candidates": 0,
        }


    # ==============================================================================================
    # CROP GEOMETRY
    # ==============================================================================================

    crop_h = int(
        crop.shape[-2]
    )

    crop_w = int(
        crop.shape[-1]
    )

    max_x = (
        canonical_size
        -
        crop_w
    )

    max_y = (
        canonical_size
        -
        crop_h
    )


    # ==============================================================================================
    # GENERATE PHASE-REFINED PLACEMENTS
    #
    # IMPORTANT:
    # These offsets are generated blindly.
    # No true XY is used.
    # ==============================================================================================

    phase_offsets = [
        -phase_step,
        0,
        phase_step,
    ]


    placement_map = {}


    for base_rank, base in enumerate(
        candidates[:top_k],
        start=1,
    ):

        base_x = int(
            base["x"]
        )

        base_y = int(
            base["y"]
        )


        for dy in phase_offsets:

            for dx in phase_offsets:

                x = (
                    base_x
                    +
                    dx
                )

                y = (
                    base_y
                    +
                    dy
                )


                # ----------------------------------------------------------------------------------
                # Physical validity
                # ----------------------------------------------------------------------------------

                if x < 0 or y < 0:
                    continue

                if x > max_x or y > max_y:
                    continue


                key = (
                    int(x),
                    int(y),
                )


                # ----------------------------------------------------------------------------------
                # Deduplicate.
                #
                # Prefer hypothesis derived from better-ranked base candidate.
                # ----------------------------------------------------------------------------------

                if key not in placement_map:

                    placement_map[
                        key
                    ] = {
                        "x": int(x),
                        "y": int(y),
                        "base_rank": int(base_rank),
                        "base_x": int(base_x),
                        "base_y": int(base_y),
                        "dx": int(dx),
                        "dy": int(dy),
                        "base_score": base.get(
                            "score"
                        ),
                    }


    placements = list(
        placement_map.values()
    )


    # ==============================================================================================
    # Rank phase hypotheses
    #
    # Base rank first.
    # Unshifted candidate first within each base candidate.
    # Then single-axis shifts.
    # Then diagonal shifts.
    # ==============================================================================================

    def phase_priority(p):

        displacement = (
            abs(
                p["dx"]
            )
            +
            abs(
                p["dy"]
            )
        )

        return (
            p["base_rank"],
            displacement,
        )


    placements.sort(
        key=phase_priority
    )


    # ==============================================================================================
    # STRICT AUTHENTICATED RECOVERY
    # ==============================================================================================

    authenticated = []


    for placement_rank, p in enumerate(
        placements,
        start=1,
    ):

        recovery = recover_v410_at_placement(
            crop=crop,
            candidate_x=p["x"],
            candidate_y=p["y"],
        )


        if not recovery.get(
            "accepted",
            False,
        ):
            continue


        token = recovery.get(
            "token"
        )


        if token is None:
            continue


        authenticated.append(
            {
                "token": token,

                "x": p["x"],
                "y": p["y"],

                "base_rank":
                    p["base_rank"],

                "placement_rank":
                    placement_rank,

                "phase_dx":
                    p["dx"],

                "phase_dy":
                    p["dy"],

                "base_x":
                    p["base_x"],

                "base_y":
                    p["base_y"],

                "recovery":
                    recovery,
            }
        )


    # ==============================================================================================
    # NOTHING AUTHENTICATED
    # ==============================================================================================

    if len(authenticated) == 0:

        return {
            "accepted": False,
            "ambiguous": False,
            "candidate": None,

            "tested_placements":
                len(placements),

            "base_candidates":
                min(
                    top_k,
                    len(candidates),
                ),
        }


    # ==============================================================================================
    # DEDUPLICATE BY AUTHENTICATED SOURCE TOKEN
    # ==============================================================================================

    unique_by_token = {}


    for candidate in authenticated:

        token = candidate[
            "token"
        ]


        if token not in unique_by_token:

            unique_by_token[
                token
            ] = candidate


    unique = list(
        unique_by_token.values()
    )


    # ==============================================================================================
    # MULTIPLE AUTHENTICATED IDENTITIES
    # ==============================================================================================

    if len(unique) > 1:

        return {
            "accepted": False,
            "ambiguous": True,
            "candidate": None,

            "authenticated_candidates":
                unique,

            "tested_placements":
                len(placements),

            "base_candidates":
                min(
                    top_k,
                    len(candidates),
                ),
        }


    # ==============================================================================================
    # UNIQUE AUTHENTICATED SOURCE
    # ==============================================================================================

    return {
        "accepted": True,
        "ambiguous": False,

        "candidate":
            unique[0],

        "authenticated_candidates":
            unique,

        "tested_placements":
            len(placements),

        "base_candidates":
            min(
                top_k,
                len(candidates),
            ),
    }


# ====================================================================================================================
# recover_v410_at_placement
# ====================================================================================================================
@torch.no_grad()
def recover_v415_at_placement(
    crop,
    candidate_x,
    candidate_y,
):

    existing = _V415_V414_RECOVER(
        crop=crop,
        candidate_x=candidate_x,
        candidate_y=candidate_y,
    )

    if existing.get(
        "accepted",
        False,
    ):

        result = dict(existing)

        result["v415_method"] = result.get(
            "v414_method",
            result.get(
                "v413_method",
                "existing",
            ),
        )

        return result

    if existing.get("status") == "ambiguous":
        return existing

    soft = recover_v415_soft_auth_at_placement(
        crop=crop,
        candidate_x=candidate_x,
        candidate_y=candidate_y,
        source_pool_size=8,
        auth_pool_size=8,
        auth_top_k=4,
        min_auth_support=6,
    )

    if soft.get(
        "accepted",
        False,
    ):

        soft["previous_status"] = existing.get(
            "status"
        )

        return soft

    if soft.get("status") == "ambiguous":
        return soft

    result = dict(existing)

    result["v415_method"] = "unresolved"
    result["soft_auth_status"] = soft.get(
        "status"
    )

    return result


# ====================================================================================================================
# blind_crop_search_spatial_fast
# ====================================================================================================================
def blind_crop_search_spatial_fast(crop):
    """
    Blind spatial synchronization.

    Stage A:
        coarse placement search

    Stage B:
        pixel-level refinement within +/- REFINE_RADIUS
        around the top coarse candidates.
    """

    if crop.ndim == 4:

        if crop.shape[0] != 1:
            raise ValueError(
                f"Expected single crop batch, got shape {tuple(crop.shape)}"
            )

        crop3 = crop[0]

    elif crop.ndim == 3:

        crop3 = crop

    else:

        raise ValueError(
            f"Unexpected crop shape {tuple(crop.shape)}"
        )

    _, crop_h, crop_w = crop3.shape

    max_x = 256 - crop_w
    max_y = 256 - crop_h

    if max_x < 0 or max_y < 0:
        raise ValueError(
            "Crop is larger than the 256x256 canonical canvas."
        )

    xs = list(
        range(
            0,
            max_x + 1,
            COARSE_STRIDE,
        )
    )

    ys = list(
        range(
            0,
            max_y + 1,
            COARSE_STRIDE,
        )
    )

    if max_x not in xs:
        xs.append(max_x)

    if max_y not in ys:
        ys.append(max_y)

    coarse_placements = [
        (x, y)
        for y in ys
        for x in xs
    ]

    coarse_scored = _score_placement_list(
        crop3,
        coarse_placements,
    )

    coarse_ranked = _rank_spatial_candidates(
        coarse_scored
    )

    coarse_top = coarse_ranked[
        :TOP_COARSE
    ]

    refine_set = set()

    for candidate in coarse_top:

        cx = int(
            candidate["x"]
        )

        cy = int(
            candidate["y"]
        )

        x0 = max(
            0,
            cx - REFINE_RADIUS,
        )

        x1 = min(
            max_x,
            cx + REFINE_RADIUS,
        )

        y0 = max(
            0,
            cy - REFINE_RADIUS,
        )

        y1 = min(
            max_y,
            cy + REFINE_RADIUS,
        )

        for y in range(
            y0,
            y1 + 1,
        ):

            for x in range(
                x0,
                x1 + 1,
            ):

                refine_set.add(
                    (x, y)
                )

    refine_placements = sorted(
        refine_set
    )

    refined_scored = _score_placement_list(
        crop3,
        refine_placements,
    )

    refined_ranked = _rank_spatial_candidates(
        refined_scored
    )

    return {
        "ranked":
            refined_ranked,

        "coarse_ranked":
            coarse_ranked,

        "coarse_count":
            len(coarse_placements),

        "refined_count":
            len(refine_placements),
    }


# ====================================================================================================================
# _score_placement_list
# ====================================================================================================================
def _score_placement_list(
    crop,
    placements,
):
    output = []

    for start in range(
        0,
        len(placements),
        BATCH_PLACEMENTS,
    ):
        chunk = placements[
            start:
            start + BATCH_PLACEMENTS
        ]

        output.extend(
            score_spatial_index_batch(
                crop,
                chunk,
            )
        )

    return output


# ====================================================================================================================
# _rank_spatial_candidates
# ====================================================================================================================
def _rank_spatial_candidates(
    candidates,
):
    return sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )


# ====================================================================================================================
# score_spatial_index_batch
# ====================================================================================================================
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


# ====================================================================================================================
# region_visibility_weights
# ====================================================================================================================
def region_visibility_weights(
    crop_height,
    crop_width,
    candidate_x,
    candidate_y,
    canonical_size=256,
    region_size=64,
    device=None,
):

    crop_x0 = int(candidate_x)
    crop_y0 = int(candidate_y)

    crop_x1 = crop_x0 + int(crop_width)
    crop_y1 = crop_y0 + int(crop_height)

    weights = []

    for region_index in range(16):

        rr = region_index // 4
        cc = region_index % 4

        rx0 = cc * region_size
        ry0 = rr * region_size

        rx1 = rx0 + region_size
        ry1 = ry0 + region_size

        overlap_w = max(
            0,
            min(crop_x1, rx1)
            -
            max(crop_x0, rx0),
        )

        overlap_h = max(
            0,
            min(crop_y1, ry1)
            -
            max(crop_y0, ry0),
        )

        overlap_area = (
            overlap_w
            *
            overlap_h
        )

        fraction = (
            overlap_area
            /
            float(
                region_size
                *
                region_size
            )
        )

        weights.append(
            fraction
        )

    return torch.tensor(
        weights,
        device=device,
        dtype=torch.float32,
    )


# ====================================================================================================================
# split_batch_16_regions_v2
# ====================================================================================================================
def split_batch_16_regions_v2(
    images,
):

    if images.ndim != 4:
        raise ValueError(
            f"Expected [B,3,256,256], got {images.shape}"
        )

    B, C, H, W = images.shape

    if H != 256 or W != 256:
        raise ValueError(
            f"Expected 256x256 canvases, got {H}x{W}"
        )

    return (
        images
        .reshape(
            B,
            C,
            4,
            64,
            4,
            64,
        )
        .permute(
            0,
            2,
            4,
            1,
            3,
            5,
        )
        .contiguous()
        .reshape(
            B * 16,
            C,
            64,
            64,
        )
    )


# ====================================================================================================================
# embed_full_spatial_image_v4
# ====================================================================================================================
@torch.no_grad()
def embed_full_spatial_image_v4(
    clean_image,
    source_id,
):

    # ----------------------------------------------------------------------------------------------
    # V4 packet issuance
    # ----------------------------------------------------------------------------------------------

    fragments = v4_issue(
        source_id,
        V3H_TRAIN_KEY,
    )


    bits20 = fragments_to_bits20_v49(
        fragments
    )


    # ----------------------------------------------------------------------------------------------
    # Existing image tiling
    # ----------------------------------------------------------------------------------------------

    tiles = split_16_regions(
        clean_image
    ).to(
        DEVICE
    )


    # ----------------------------------------------------------------------------------------------
    # Same 20-bit payload repeated over 4x4 local packet cells
    # ----------------------------------------------------------------------------------------------

    payload20 = (
        bits20[
            :,
            None,
            None,
            :
        ]
        .expand(
            -1,
            4,
            4,
            -1,
        )
        .contiguous()
    )


    # ----------------------------------------------------------------------------------------------
    # Preserve existing 44-bit network packet interface
    # ----------------------------------------------------------------------------------------------

    packet44 = torch.zeros(
        16,
        4,
        4,
        44,
        device=DEVICE,
        dtype=payload20.dtype,
    )


    packet44[
        ...,
        :20
    ] = payload20


    # ----------------------------------------------------------------------------------------------
    # Existing watermark encoder
    # ----------------------------------------------------------------------------------------------

    output = v2_model_audit(
        tiles,
        packet44,
        amplitude=CAL_AMPLITUDE,
        active_tag_bits=CAL_ACTIVE_TAG_BITS,
    )


    watermarked_tiles = output[
        "watermarked_image"
    ]


    full_watermarked = assemble_16_regions(
        watermarked_tiles
    )


    return {
        "watermarked":
            full_watermarked,

        "watermarked_tiles":
            watermarked_tiles,

        "target_bits20":
            bits20,

        "fragments":
            fragments,

        "version":
            "V4",
    }


# ====================================================================================================================
# v4_source_token
# ====================================================================================================================
def v4_source_token(
    full_source_id,
    key,
):

    full_source_id = bytes(
        full_source_id
    )

    digest = hmac.new(
        key,
        V4_SOURCE_DOMAIN
        + b"\x00"
        + full_source_id,
        hashlib.sha256,
    ).digest()

    return digest[
        :V4_TOKEN_BYTES
    ]


# ====================================================================================================================
# v4_compute_auth_tag
# ====================================================================================================================
def v4_compute_auth_tag(
    token48,
    key,
):

    token48 = bytes(
        token48
    )

    if len(token48) != V4_TOKEN_BYTES:
        raise ValueError(
            f"Expected {V4_TOKEN_BYTES}-byte V4 token."
        )

    digest = hmac.new(
        key,
        V4_AUTH_DOMAIN
        + b"\x00"
        + token48,
        hashlib.sha256,
    ).digest()

    return digest[
        :V4_AUTH_BYTES
    ]


# ====================================================================================================================
# v4_issue
# ====================================================================================================================
def v4_issue(
    full_source_id,
    key,
):

    token48 = v4_source_token(
        full_source_id,
        key,
    )

    auth48 = v4_compute_auth_tag(
        token48,
        key,
    )


    source_symbols = bytes(
        V4_RS_CODEC.encode(
            token48
        )
    )

    auth_symbols = bytes(
        V4_RS_CODEC.encode(
            auth48
        )
    )


    if len(source_symbols) != 16:
        raise RuntimeError(
            f"V4 source codeword length = {len(source_symbols)}, expected 16"
        )

    if len(auth_symbols) != 16:
        raise RuntimeError(
            f"V4 auth codeword length = {len(auth_symbols)}, expected 16"
        )


    return [
        (
            i,
            int(source_symbols[i]),
            int(auth_symbols[i]),
        )
        for i in range(16)
    ]


# ====================================================================================================================
# v4_rs_decode
# ====================================================================================================================
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


# ====================================================================================================================
# v4_try_rs_configuration
# ====================================================================================================================
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

