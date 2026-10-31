import torch


@torch.no_grad()
def extract_raw_logits20(tiles, runtime):
    device = runtime["device"]
    decoder = runtime["v3_independent_decoder"]

    raw = decoder(
        tiles.to(device)
    )

    if "auth_logits" in raw:
        auth = raw["auth_logits"]

    elif "auth8_logits" in raw:
        auth = raw["auth8_logits"]

    elif "auth32_logits" in raw:
        auth = raw["auth32_logits"][..., :8]

    else:
        raise RuntimeError(
            "Could not locate auth logits."
        )

    raw20 = torch.cat(
        [
            raw["index_logits"],
            raw["source_logits"],
            auth,
        ],
        dim=-1,
    )

    if raw20.shape[-1] != 20:
        raise RuntimeError(
            f"Expected 20 raw logits, got {tuple(raw20.shape)}"
        )

    return raw20


@torch.no_grad()
def old_fusion_logits20(raw_logits20, runtime):
    fusion_model = runtime["fusion_model"]

    fused = fusion_model(
        raw_logits20
    )

    if isinstance(fused, dict):
        if "logits20" not in fused:
            raise RuntimeError(
                "fusion_model returned dict without logits20"
            )

        return fused["logits20"]

    if torch.is_tensor(fused):
        return fused

    raise TypeError(
        "Unexpected fusion_model output: "
        f"{type(fused).__name__}"
    )


@torch.no_grad()
def decode_phase_tiles_kfrag_final_100k(
    tiles,
    runtime,
):
    device = runtime["device"]
    verification_model = runtime["verification_model"]

    tiles = tiles.to(device)

    raw_logits20 = extract_raw_logits20(
        tiles,
        runtime,
    )

    base_fused = old_fusion_logits20(
        raw_logits20,
        runtime,
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


@torch.no_grad()
def frozen_fragment_features(
    regions,
    visibility,
    runtime,
):
    device = runtime["device"]
    verification_model = runtime["verification_model"]

    regions = regions.to(device)

    raw = extract_raw_logits20(
        regions,
        runtime,
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
        raw,
        runtime,
    )

    refined = verification_model(
        regions,
        raw,
    )["fragment_logits20"]

    final = (
        base
        + refined
        - raw_mean
    )

    visibility = (
        visibility
        .reshape(16, 1)
        .float()
        .to(device)
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

    if features.shape != (16, 141):
        raise RuntimeError(
            f"Expected fragment features [16,141], got {tuple(features.shape)}"
        )

    return {
        "raw": raw,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "raw_max": raw_max,
        "raw_min": raw_min,
        "base": base,
        "refined": refined,
        "final": final,
        "visibility": visibility,
        "features": features,
    }

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
            min(crop_x1, rx1) - max(crop_x0, rx0),
        )

        overlap_h = max(
            0,
            min(crop_y1, ry1) - max(crop_y0, ry0),
        )

        overlap_area = overlap_w * overlap_h

        fraction = (
            overlap_area
            / float(region_size * region_size)
        )

        weights.append(fraction)

    return torch.tensor(
        weights,
        device=device,
        dtype=torch.float32,
    )


def split_batch_16_regions_v2(images):
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

def _batch_candidate_canvases(
    crop,
    placements,
    runtime,
    canonical_size=256,
    neutral_value=0.5,
):
    device = runtime["device"]

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

    crop3 = crop3.to(device)

    C, H, W = crop3.shape

    canvases = torch.full(
        (
            len(placements),
            C,
            canonical_size,
            canonical_size,
        ),
        float(neutral_value),
        device=crop3.device,
        dtype=crop3.dtype,
    )

    for i, (x, y) in enumerate(placements):
        x = int(x)
        y = int(y)

        if (
            x < 0
            or y < 0
            or x + W > canonical_size
            or y + H > canonical_size
        ):
            raise ValueError(
                f"Invalid placement {(x, y)} for crop {(H, W)}"
            )

        canvases[
            i,
            :,
            y:y + H,
            x:x + W,
        ] = crop3

    return canvases

def canonical_index_bits(
    device=None,
    dtype=torch.float32,
):
    values = torch.arange(
        16,
        device=device,
        dtype=torch.long,
    )

    shifts = torch.tensor(
        [3, 2, 1, 0],
        device=device,
        dtype=torch.long,
    )

    bits = (
        (
            values[:, None]
            >> shifts[None, :]
        )
        & 1
    )

    return bits.to(dtype=dtype)


@torch.no_grad()
def decode_phase_tiles_v2(
    tiles,
    runtime,
):
    return decode_phase_tiles_kfrag_final_100k(
        tiles,
        runtime,
    )

@torch.no_grad()
def _score_spatial_index_batch(
    crop,
    placements,
    runtime,
):
    if len(placements) == 0:
        return []

    crop3 = (
        crop[0]
        if crop.ndim == 4
        else crop
    )

    _, crop_h, crop_w = crop3.shape

    canvases = _batch_candidate_canvases(
        crop=crop3,
        placements=placements,
        runtime=runtime,
    )

    tiles = split_batch_16_regions_v2(
        canvases
    )

    decoded = decode_phase_tiles_v2(
        tiles,
        runtime,
    )

    if isinstance(decoded, dict):
        logits20 = decoded[
            "final_logits20"
        ]

    elif torch.is_tensor(decoded):
        logits20 = decoded

    else:
        raise TypeError(
            "decode_phase_tiles_v2 returned unsupported type: "
            f"{type(decoded).__name__}"
        )

    B = len(placements)

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
            device=index_probs.device,
            dtype=index_probs.dtype,
        )
        .unsqueeze(0)
    )

    # Probability assigned to the correct index bit.
    bit_match_prob = (
        index_probs
        * target_bits
        +
        (
            1.0
            - index_probs
        )
        *
        (
            1.0
            - target_bits
        )
    )

    # Region-level probabilistic index agreement.
    region_prob_score = (
        bit_match_prob.mean(
            dim=-1
        )
    )

    # Hard exact 4-bit region index match.
    hard_bits = (
        index_probs
        >= 0.5
    ).to(
        target_bits.dtype
    )

    hard_region_match = (
        hard_bits
        == target_bits
    ).all(
        dim=-1
    ).float()

    # Index confidence.
    bit_confidence = (
        torch.abs(
            index_probs
            - 0.5
        )
        * 2.0
    )

    region_confidence = (
        bit_confidence.mean(
            dim=-1
        )
    )

    results = []

    for i, (x, y) in enumerate(placements):

        visibility = region_visibility_weights(
            crop_height=crop_h,
            crop_width=crop_w,
            candidate_x=x,
            candidate_y=y,
            device=index_probs.device,
        )

        denom = (
            visibility.sum()
            + 1e-8
        )

        protocol_score = (
            (
                region_prob_score[i]
                * visibility
            ).sum()
            / denom
        )

        hard_alignment = (
            (
                hard_region_match[i]
                * visibility
            ).sum()
            / denom
        )

        confidence = (
            (
                region_confidence[i]
                * visibility
            ).sum()
            / denom
        )

        results.append(
            {
                "x": int(x),
                "y": int(y),

                "score": float(
                    protocol_score.item()
                ),

                "hard_alignment": float(
                    hard_alignment.item()
                ),

                "confidence": float(
                    confidence.item()
                ),
            }
        )

    return results

BATCH_PLACEMENTS = 24
COARSE_STRIDE = 4
TOP_COARSE = 20
REFINE_RADIUS = 6


def _score_placement_list(
    crop,
    placements,
    runtime,
):
    output = []

    device = runtime["device"]

    # Preserve the frozen GPU setting.
    # Use a smaller batch on CPU to avoid processing
    # hundreds of region tiles simultaneously.
    if device.type == "cuda":
        batch_placements = 24
    else:
        batch_placements = 2

    for start in range(
        0,
        len(placements),
        batch_placements,
    ):
        chunk = placements[
            start:
            start + batch_placements
        ]

        output.extend(
            _score_spatial_index_batch(
                crop,
                chunk,
                runtime,
            )
        )

    return output

def _rank_spatial_candidates(
    candidates,
):
    return sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

def blind_crop_search_spatial_fast(
    crop,
    runtime,
):
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

    # --------------------------------------------------------
    # Stage A: coarse placement search
    # --------------------------------------------------------

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
        runtime,
    )

    coarse_ranked = _rank_spatial_candidates(
        coarse_scored
    )

    coarse_top = coarse_ranked[
        :TOP_COARSE
    ]

    # --------------------------------------------------------
    # Stage B: pixel-level refinement
    # --------------------------------------------------------

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
        runtime,
    )

    refined_ranked = _rank_spatial_candidates(
        refined_scored
    )

    return {
        "ranked": refined_ranked,
        "coarse_ranked": coarse_ranked,
        "coarse_count": len(coarse_placements),
        "refined_count": len(refine_placements),
    }