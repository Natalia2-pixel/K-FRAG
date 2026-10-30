@torch.no_grad()
def make_weak_symbol_sample(
    path,
    area,
    rng,
):

    clean = load_eval_image(
        path
    ).to(
        DEVICE
    )


    if clean.ndim == 3:

        clean = clean.unsqueeze(
            0
        )


    source_id = evaluation_source_id(
        "KFRAG_CV_TRAIN",
        Path(path).name,
    )


    embedded = embed_full_spatial_image(
        clean,
        source_id,
    )


    watermarked = embedded[
        "watermarked"
    ]


    target_bits20 = normalize_target_bits(
        embedded[
            "target_bits20"
        ]
    )


    crop, metadata = make_physical_crop(
        image=watermarked,
        crop_type="random_rectangle",
        retain=math.sqrt(
            area
        ),
        rng=rng,
    )


    x = int(
        metadata[
            "x"
        ]
    )

    y = int(
        metadata[
            "y"
        ]
    )


    canvas = place_crop_on_canvas(
        crop=crop,
        candidate_x=x,
        candidate_y=y,
        canonical_size=256,
        neutral_value=0.5,
    )


    regions = split_batch_16_regions_v2(
        canvas
    ).to(
        DEVICE
    )


    visibility = region_visibility_weights(
        crop_height=int(
            crop.shape[-2]
        ),
        crop_width=int(
            crop.shape[-1]
        ),
        candidate_x=x,
        candidate_y=y,
        device=DEVICE,
    )


    if not torch.is_tensor(
        visibility
    ):

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


    decoder = frozen_fragment_features(
        regions,
        visibility,
    )


    true_source_byte = bits8_to_byte(
        target_bits20[
            :,
            4:12
        ]
    )


    baseline_byte_scores = source_byte_scores_local(
        decoder[
            "final"
        ]
    )


    return {
        "features":
            decoder[
                "features"
            ].detach(),

        "baseline_byte_scores":
            baseline_byte_scores.detach(),

        "target_byte":
            true_source_byte.detach(),

        "visibility":
            visibility.detach(),

        "final_logits20":
            decoder[
                "final"
            ].detach(),

        "source_id":
            source_id,
    }
