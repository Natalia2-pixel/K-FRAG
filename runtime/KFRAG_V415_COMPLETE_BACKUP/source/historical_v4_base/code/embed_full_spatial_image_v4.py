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
