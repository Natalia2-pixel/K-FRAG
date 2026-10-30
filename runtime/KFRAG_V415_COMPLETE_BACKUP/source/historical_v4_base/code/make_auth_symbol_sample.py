@torch.no_grad()
def make_auth_symbol_sample(path, area, rng):

    clean = load_eval_image(
        path
    ).to(DEVICE)

    if clean.ndim == 3:
        clean = clean.unsqueeze(0)


    source_id = evaluation_source_id(
        "KFRAG_CV_TRAIN",
        Path(path).name,
    )


    embedded = embed_full_spatial_image(
        clean,
        source_id,
    )

    watermarked = embedded["watermarked"]

    target_bits20 = normalize_target_bits(
        embedded["target_bits20"]
    )


    crop_sample, metadata = make_physical_crop(
        image=watermarked,
        crop_type="random_rectangle",
        retain=math.sqrt(area),
        rng=rng,
    )


    x = int(metadata["x"])
    y = int(metadata["y"])


    canvas = place_crop_on_canvas(
        crop=crop_sample,
        candidate_x=x,
        candidate_y=y,
        canonical_size=256,
        neutral_value=0.5,
    )


    regions = split_batch_16_regions_v2(
        canvas
    ).to(DEVICE)


    visibility = region_visibility_weights(
        crop_height=int(crop_sample.shape[-2]),
        crop_width=int(crop_sample.shape[-1]),
        candidate_x=x,
        candidate_y=y,
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


    decoder = frozen_fragment_features(
        regions,
        visibility,
    )


    final_logits20 = decoder["final"]


    # ----------------------------------------------------------------------------------------------
    # AUTH TARGET
    # ----------------------------------------------------------------------------------------------

    true_auth_byte = bits8_to_byte(
        target_bits20[:, 12:20]
    )


    # ----------------------------------------------------------------------------------------------
    # CORRECT AUTH BASELINE BYTE SCORES
    # ----------------------------------------------------------------------------------------------

    auth_baseline_byte_scores = auth_byte_scores_local(
        final_logits20
    )


    return {
        "features":
            decoder["features"].detach(),

        "baseline_byte_scores":
            auth_baseline_byte_scores.detach(),

        "target_byte":
            true_auth_byte.detach(),

        "visibility":
            visibility.detach(),

        "final_logits20":
            final_logits20.detach(),

        "source_id":
            source_id,
    }
