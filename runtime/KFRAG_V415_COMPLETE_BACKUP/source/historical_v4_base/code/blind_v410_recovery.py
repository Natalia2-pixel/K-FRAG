@torch.no_grad()
def blind_v410_recovery(
    crop,
    max_placements=20,
):

    search_result = blind_crop_search_spatial_fast(
        crop
    )


    candidates = normalize_blind_candidates(
        search_result
    )


    candidates = candidates[
        :int(max_placements)
    ]


    evaluated = []
    authenticated = []


    for spatial_rank, placement in enumerate(
        candidates,
        start=1,
    ):

        result = recover_v410_at_placement(
            crop=crop,
            candidate_x=int(
                placement["x"]
            ),
            candidate_y=int(
                placement["y"]
            ),
        )


        result[
            "spatial_rank"
        ] = spatial_rank


        result[
            "search_score"
        ] = placement.get(
            "search_score",
            placement.get(
                "score",
                None,
            ),
        )


        evaluated.append(
            result
        )


        if result[
            "accepted"
        ]:

            authenticated.append(
                result
            )


    # ----------------------------------------------------------------------------------------------
    # Deduplicate same token across nearby XY aliases
    # ----------------------------------------------------------------------------------------------

    token_best = {}


    for item in authenticated:

        token = item[
            "token"
        ]

        previous = token_best.get(
            token
        )


        if (
            previous is None
            or
            item.get(
                "support",
                -1e9,
            )
            >
            previous.get(
                "support",
                -1e9,
            )
        ):

            token_best[
                token
            ] = item


    unique_candidates = list(
        token_best.values()
    )


    unique_candidates.sort(
        key=lambda item:
            (
                item.get(
                    "support",
                    -1e9,
                ),
                -item.get(
                    "spatial_rank",
                    999999,
                ),
            ),
        reverse=True,
    )


    return {
        "accepted":
            len(
                unique_candidates
            ) > 0,

        "ambiguous":
            len(
                unique_candidates
            ) > 1,

        "candidate":
            (
                unique_candidates[0]
                if unique_candidates
                else None
            ),

        "unique_candidates":
            unique_candidates,

        "evaluated":
            evaluated,

        "spatial_candidates":
            candidates,
    }
