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
