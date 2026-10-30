# ================================================================================
# K-FRAG V4.15 ACTIVE RUNTIME SOURCE SNAPSHOT
# Generated: 2026-10-26T23:43:10.413029Z
# ================================================================================

import hmac
import hashlib
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations



# ==============================================================================================================
# auth_byte_scores_local
# ==============================================================================================================

@torch.no_grad()
def auth_byte_scores_local(final_logits20):

    if tuple(final_logits20.shape) != (16, 20):
        raise RuntimeError(
            f"Expected final_logits20=(16,20), got {tuple(final_logits20.shape)}"
        )

    # source_byte_scores_local() expects the source field at [4:12].
    # Copy auth logits [12:20] into those positions.
    proxy = final_logits20.clone()

    proxy[:, 4:12] = final_logits20[:, 12:20]

    scores = source_byte_scores_local(proxy)

    if tuple(scores.shape) != (16, 256):
        raise RuntimeError(
            f"Expected AUTH byte scores=(16,256), got {tuple(scores.shape)}"
        )

    return scores



# ==============================================================================================================
# bits8_to_byte
# ==============================================================================================================

def bits8_to_byte(bits8):

    bits8 = (
        bits8
        .to(torch.int64)
    )

    weights = torch.tensor(
        [
            128,
            64,
            32,
            16,
            8,
            4,
            2,
            1,
        ],
        dtype=torch.int64,
        device=bits8.device,
    )

    return (
        bits8
        *
        weights[None, :]
    ).sum(
        dim=1
    )



# ==============================================================================================================
# blind_authenticated_recovery
# ==============================================================================================================

def blind_authenticated_recovery(
    observations,
    key,
):

    started = time.perf_counter()


    (
        candidates,
        usable_subset_count,
    ) = generate_blind_source_candidates(
        observations
    )


    verified = []


    for candidate_source, candidate_info in candidates.items():

        verification = verify_candidate_strict(
            candidate_source,
            observations,
            key,
        )


        if (
            verification[
                "valid_count"
            ]
            >=
            STRICT_VERIFY_COUNT
        ):

            verified.append(
                {
                    "source":
                        candidate_source,

                    "valid_count":
                        verification[
                            "valid_count"
                        ],

                    "matching_physical_regions":
                        verification[
                            "matching_physical_regions"
                        ],

                    "supporting_subsets":
                        candidate_info[
                            "supporting_subsets"
                        ],
                }
            )


    # Highest strict authenticated support first.
    verified.sort(
        key=lambda x: (
            x[
                "valid_count"
            ],
            x[
                "supporting_subsets"
            ],
        ),
        reverse=True,
    )


    if len(
        verified
    ) == 0:

        status = "unresolved"

        recovered_source = None

        valid_count = 0


    elif len(
        verified
    ) == 1:

        status = "accepted"

        recovered_source = verified[
            0
        ][
            "source"
        ]

        valid_count = verified[
            0
        ][
            "valid_count"
        ]


    else:

        # Multiple independently authenticated identities
        # should not silently resolve.
        status = "ambiguous"

        recovered_source = None

        valid_count = verified[
            0
        ][
            "valid_count"
        ]


    elapsed = (
        time.perf_counter()
        -
        started
    )


    return {
        "status":
            status,

        "source":
            recovered_source,

        "valid_count":
            valid_count,

        "verified_candidates":
            verified,

        "candidate_count":
            len(
                candidates
            ),

        "usable_subset_count":
            usable_subset_count,

        "runtime_s":
            elapsed,
    }



# ==============================================================================================================
# blind_crop_recovery_for_phase
# ==============================================================================================================

def blind_crop_recovery_for_phase(
    observations,
    key,
):

    candidates = generate_crop_source_candidates(
        observations
    )


    verified = []


    for source, info in candidates.items():

        verification = verify_candidate_strict(
            source,
            observations,
            key,
        )


        if (
            verification[
                "valid_count"
            ]
            >=
            8
        ):

            verified.append(
                {
                    "source":
                        source,

                    "valid_count":
                        verification[
                            "valid_count"
                        ],

                    "supporting_subsets":
                        info[
                            "supporting_subsets"
                        ],

                    "authenticated_indices":
                        verification.get(
                            "authenticated_indices",
                            [],
                        ),
                }
            )


    verified.sort(
        key=lambda item:
            (
                item[
                    "valid_count"
                ],
                item[
                    "supporting_subsets"
                ],
            ),
        reverse=True,
    )


    if len(
        verified
    ) == 0:

        return {
            "status":
                "unresolved",

            "source":
                None,

            "valid_count":
                0,
        }


    if len(
        verified
    ) > 1:

        return {
            "status":
                "ambiguous",

            "source":
                None,

            "valid_count":
                verified[
                    0
                ][
                    "valid_count"
                ],
        }


    return {
        "status":
            "accepted",

        "source":
            verified[
                0
            ][
                "source"
            ],

        "valid_count":
            verified[
                0
            ][
                "valid_count"
            ],
    }



# ==============================================================================================================
# blind_crop_synchronize_and_recover
# ==============================================================================================================

@torch.no_grad()
def blind_crop_synchronize_and_recover(
    crop,
    key,
):

    started = time.perf_counter()


    search = blind_phase_search(
        crop
    )


    phase_candidates = search[
        "ranked"
    ][
        :TOP_VERIFY_PHASES
    ]


    evaluated = []


    for candidate in phase_candidates:

        details = score_phase(
            crop,
            candidate[
                "phase_x"
            ],
            candidate[
                "phase_y"
            ],
            return_details=True,
        )


        observations = (
            phase_details_to_observations(
                details
            )
        )


        result = blind_crop_recovery_for_phase(
            observations,
            key,
        )


        evaluated.append(
            {
                "phase_x":
                    candidate[
                        "phase_x"
                    ],

                "phase_y":
                    candidate[
                        "phase_y"
                    ],

                "sync_score":
                    candidate[
                        "score"
                    ],

                "status":
                    result[
                        "status"
                    ],

                "source":
                    result[
                        "source"
                    ],

                "valid_count":
                    result[
                        "valid_count"
                    ],

                "observations":
                    len(
                        observations
                    ),
            }
        )


    # --------------------------------------------------------
    # Prefer authenticated synchronization candidates.
    # --------------------------------------------------------

    accepted = [
        item
        for item in evaluated
        if item[
            "status"
        ] == "accepted"
    ]


    if len(
        accepted
    ) > 0:

        accepted.sort(
            key=lambda item:
                (
                    item[
                        "valid_count"
                    ],
                    item[
                        "sync_score"
                    ],
                ),
            reverse=True,
        )


        best = accepted[
            0
        ]


        # Multiple different authenticated sources = ambiguity.
        accepted_sources = {
            item[
                "source"
            ]
            for item in accepted
        }


        if len(
            accepted_sources
        ) > 1:

            final_status = "ambiguous"

            final_source = None


        else:

            final_status = "accepted"

            final_source = best[
                "source"
            ]


    else:

        best = evaluated[
            0
        ]

        final_status = "unresolved"

        final_source = None


    runtime = (
        time.perf_counter()
        -
        started
    )


    return {
        "status":
            final_status,

        "source":
            final_source,

        "phase_x":
            best[
                "phase_x"
            ],

        "phase_y":
            best[
                "phase_y"
            ],

        "sync_score":
            best[
                "sync_score"
            ],

        "valid_count":
            best[
                "valid_count"
            ],

        "runtime_s":
            runtime,

        "search":
            search,

        "evaluated":
            evaluated,
    }



# ==============================================================================================================
# blind_irregular_phase_search
# ==============================================================================================================

@torch.no_grad()
def blind_irregular_phase_search(
    questioned,
    questioned_mask,
):

    # --------------------------------------------------------
    # COARSE
    # --------------------------------------------------------

    coarse = []


    for phase_y in range(
        0,
        64,
        IRREGULAR_COARSE_STEP,
    ):

        for phase_x in range(
            0,
            64,
            IRREGULAR_COARSE_STEP,
        ):

            coarse.append(
                score_irregular_phase(
                    questioned,
                    questioned_mask,
                    phase_x,
                    phase_y,
                )
            )


    coarse.sort(
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    # --------------------------------------------------------
    # LOCAL REFINE
    # --------------------------------------------------------

    refine_pairs = set()


    for candidate in coarse[
        :IRREGULAR_TOP_COARSE
    ]:

        for dy in range(
            -IRREGULAR_REFINE_RADIUS,
            IRREGULAR_REFINE_RADIUS + 1,
            IRREGULAR_REFINE_STEP,
        ):

            for dx in range(
                -IRREGULAR_REFINE_RADIUS,
                IRREGULAR_REFINE_RADIUS + 1,
                IRREGULAR_REFINE_STEP,
            ):

                refine_pairs.add(
                    (
                        (
                            candidate[
                                "phase_x"
                            ]
                            +
                            dx
                        )
                        %
                        64,

                        (
                            candidate[
                                "phase_y"
                            ]
                            +
                            dy
                        )
                        %
                        64,
                    )
                )


    refined = []


    for px, py in refine_pairs:

        refined.append(
            score_irregular_phase(
                questioned,
                questioned_mask,
                px,
                py,
            )
        )


    refined.sort(
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    # --------------------------------------------------------
    # INTEGER FINAL REFINE
    # --------------------------------------------------------

    best_refined = refined[
        0
    ]


    final_pairs = set()


    for dy in range(
        -IRREGULAR_FINAL_RADIUS,
        IRREGULAR_FINAL_RADIUS + 1,
    ):

        for dx in range(
            -IRREGULAR_FINAL_RADIUS,
            IRREGULAR_FINAL_RADIUS + 1,
        ):

            final_pairs.add(
                (
                    (
                        best_refined[
                            "phase_x"
                        ]
                        +
                        dx
                    )
                    %
                    64,

                    (
                        best_refined[
                            "phase_y"
                        ]
                        +
                        dy
                    )
                    %
                    64,
                )
            )


    final_results = []


    for px, py in final_pairs:

        final_results.append(
            score_irregular_phase(
                questioned,
                questioned_mask,
                px,
                py,
            )
        )


    combined = {}


    for result in (
        refined
        +
        final_results
    ):

        key = (
            result[
                "phase_x"
            ],
            result[
                "phase_y"
            ],
        )


        if (
            key not in combined
            or
            result[
                "score"
            ]
            >
            combined[
                key
            ][
                "score"
            ]
        ):

            combined[
                key
            ] = result


    ranked = sorted(
        combined.values(),
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    return {
        "coarse":
            coarse,

        "ranked":
            ranked,
    }



# ==============================================================================================================
# blind_irregular_recover
# ==============================================================================================================

@torch.no_grad()
def blind_irregular_recover(
    questioned,
    questioned_mask,
    key,
):

    started = time.perf_counter()


    search = blind_irregular_phase_search(
        questioned,
        questioned_mask,
    )


    phase_candidates = search[
        "ranked"
    ][
        :IRREGULAR_TOP_VERIFY_PHASES
    ]


    evaluated = []


    for candidate in phase_candidates:

        details = score_irregular_phase(
            questioned,
            questioned_mask,
            candidate[
                "phase_x"
            ],
            candidate[
                "phase_y"
            ],
            return_details=True,
        )


        observations = irregular_details_to_observations(
            details
        )


        recovery = recover_irregular_phase(
            observations,
            key,
        )


        evaluated.append(
            {
                "phase_x":
                    candidate[
                        "phase_x"
                    ],

                "phase_y":
                    candidate[
                        "phase_y"
                    ],

                "sync_score":
                    candidate[
                        "score"
                    ],

                "status":
                    recovery[
                        "status"
                    ],

                "source":
                    recovery[
                        "source"
                    ],

                "valid_count":
                    recovery[
                        "valid_count"
                    ],

                "observations":
                    len(
                        observations
                    ),
            }
        )


    accepted = [
        item
        for item in evaluated
        if item[
            "status"
        ] == "accepted"
    ]


    if len(
        accepted
    ) == 0:

        best = evaluated[
            0
        ]


        status = "unresolved"

        source = None


    else:

        accepted.sort(
            key=lambda item:
                (
                    item[
                        "valid_count"
                    ],
                    item[
                        "sync_score"
                    ],
                ),
            reverse=True,
        )


        best = accepted[
            0
        ]


        accepted_sources = {
            item[
                "source"
            ]
            for item in accepted
        }


        if len(
            accepted_sources
        ) > 1:

            status = "ambiguous"

            source = None


        else:

            status = "accepted"

            source = best[
                "source"
            ]


    return {
        "status":
            status,

        "source":
            source,

        "phase_x":
            best[
                "phase_x"
            ],

        "phase_y":
            best[
                "phase_y"
            ],

        "valid_count":
            best[
                "valid_count"
            ],

        "sync_score":
            best[
                "sync_score"
            ],

        "runtime_s":
            (
                time.perf_counter()
                -
                started
            ),

        "search":
            search,

        "evaluated":
            evaluated,
    }



# ==============================================================================================================
# blind_phase_search
# ==============================================================================================================

@torch.no_grad()
def blind_phase_search(
    crop,
):

    # --------------------------------------------------------
    # A. COARSE
    # --------------------------------------------------------

    coarse_results = []


    for phase_y in range(
        0,
        64,
        COARSE_STEP,
    ):

        for phase_x in range(
            0,
            64,
            COARSE_STEP,
        ):

            coarse_results.append(
                score_phase(
                    crop,
                    phase_x,
                    phase_y,
                )
            )


    coarse_results.sort(
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    top_coarse = coarse_results[
        :TOP_COARSE
    ]


    # --------------------------------------------------------
    # B. REFINE
    # --------------------------------------------------------

    refined_phase_pairs = set()


    for candidate in top_coarse:

        center_x = candidate[
            "phase_x"
        ]

        center_y = candidate[
            "phase_y"
        ]


        for dy in range(
            -REFINE_RADIUS,
            REFINE_RADIUS + 1,
            REFINE_STEP,
        ):

            for dx in range(
                -REFINE_RADIUS,
                REFINE_RADIUS + 1,
                REFINE_STEP,
            ):

                px = (
                    center_x
                    +
                    dx
                ) % 64

                py = (
                    center_y
                    +
                    dy
                ) % 64


                refined_phase_pairs.add(
                    (
                        px,
                        py,
                    )
                )


    refined_results = []


    for (
        phase_x,
        phase_y,
    ) in refined_phase_pairs:

        refined_results.append(
            score_phase(
                crop,
                phase_x,
                phase_y,
            )
        )


    refined_results.sort(
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    # --------------------------------------------------------
    # C. FINAL INTEGER SEARCH
    # --------------------------------------------------------

    best_refined = refined_results[
        0
    ]


    final_pairs = set()


    for dy in range(
        -FINAL_RADIUS,
        FINAL_RADIUS + 1,
    ):

        for dx in range(
            -FINAL_RADIUS,
            FINAL_RADIUS + 1,
        ):

            px = (
                best_refined[
                    "phase_x"
                ]
                +
                dx
            ) % 64

            py = (
                best_refined[
                    "phase_y"
                ]
                +
                dy
            ) % 64


            final_pairs.add(
                (
                    px,
                    py,
                )
            )


    final_results = []


    for (
        phase_x,
        phase_y,
    ) in final_pairs:

        final_results.append(
            score_phase(
                crop,
                phase_x,
                phase_y,
            )
        )


    # --------------------------------------------------------
    # Combine candidates.
    # --------------------------------------------------------

    all_candidates = {}


    for result in (
        refined_results
        +
        final_results
    ):

        key = (
            result[
                "phase_x"
            ],
            result[
                "phase_y"
            ],
        )


        if (
            key not in all_candidates
            or
            result[
                "score"
            ]
            >
            all_candidates[
                key
            ][
                "score"
            ]
        ):

            all_candidates[
                key
            ] = result


    ranked = sorted(
        all_candidates.values(),
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )


    return {
        "coarse":
            coarse_results,

        "ranked":
            ranked,
    }



# ==============================================================================================================
# blind_v410_recovery
# ==============================================================================================================

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



# ==============================================================================================================
# blind_v412_phase_recovery
# ==============================================================================================================

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



# ==============================================================================================================
# decode_phase_tiles
# ==============================================================================================================

@torch.no_grad()
def decode_phase_tiles(
    tiles,
):

    raw = v3_independent_decoder(
        tiles
    )


    if "auth_logits" in raw:

        auth = raw[
            "auth_logits"
        ]

    elif "auth8_logits" in raw:

        auth = raw[
            "auth8_logits"
        ]

    elif "auth32_logits" in raw:

        auth = raw[
            "auth32_logits"
        ][
            ...,
            :8
        ]

    else:

        raise RuntimeError(
            "Could not locate auth logits."
        )


    raw20 = torch.cat(
        [
            raw[
                "index_logits"
            ],
            raw[
                "source_logits"
            ],
            auth,
        ],
        dim=-1,
    )


    fused = fusion_model(
        raw20
    )


    fused20 = fused[
        "logits20"
    ]


    # --------------------------------------------------------
    # INTERNAL HARD-PREDICTION AGREEMENT
    #
    # signed prediction is -1 / +1.
    #
    # abs(mean signed prediction)):
    #   1.0 -> all 16 cells agree
    #   0.0 -> evenly split
    # --------------------------------------------------------

    signed = torch.where(
        raw20
        >
        0,
        torch.ones_like(
            raw20
        ),
        -torch.ones_like(
            raw20
        ),
    )


    internal_agreement = (
        signed
        .mean(
            dim=(
                1,
                2,
            )
        )
        .abs()
        .mean(
            dim=1
        )
    )


    # --------------------------------------------------------
    # FUSED BIT CONFIDENCE
    # --------------------------------------------------------

    probabilities = torch.sigmoid(
        fused20
    )


    fused_confidence = (
        (
            probabilities
            -
            0.5
        )
        .abs()
        *
        2.0
    ).mean(
        dim=1
    )


    return {
        "raw20":
            raw20,

        "fused20":
            fused20,

        "internal_agreement":
            internal_agreement,

        "fused_confidence":
            fused_confidence,
    }



# ==============================================================================================================
# decode_phase_tiles_kfrag_final_100k
# ==============================================================================================================

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



# ==============================================================================================================
# decode_phase_tiles_v2
# ==============================================================================================================

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



# ==============================================================================================================
# embed_full_spatial_image
# ==============================================================================================================

@torch.no_grad()
def embed_full_spatial_image(
    clean_image,
    source_id,
):

    fragments = v3_issue(
        source_id,
        V3H_TRAIN_KEY,
    )


    bits20 = (
        v3_fragments_to_bits20(
            fragments
        )
        .reshape(
            16,
            20,
        )
        .to(
            DEVICE
        )
    )


    tiles = split_16_regions(
        clean_image
    ).to(
        DEVICE
    )


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
    }



# ==============================================================================================================
# embed_full_spatial_image_v4
# ==============================================================================================================

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



# ==============================================================================================================
# evaluation_source_id
# ==============================================================================================================

def evaluation_source_id(
    dataset_name,
    relative_identifier,
):

    message = (
        f"{EVAL_VERSION}|"
        f"{dataset_name}|"
        f"{relative_identifier}"
    ).encode(
        "utf-8"
    )


    return hashlib.sha256(
        message
    ).digest()[
        :8
    ]



# ==============================================================================================================
# fragments_to_bits20_v49
# ==============================================================================================================

def fragments_to_bits20_v49(
    fragments,
):

    if len(fragments) != 16:
        raise ValueError(
            f"Expected 16 fragments, got {len(fragments)}"
        )

    rows = []

    for item in fragments:

        if len(item) != 3:
            raise ValueError(
                "Each fragment must be (index, source_byte, auth_byte)."
            )

        index_value = int(item[0])
        source_value = int(item[1])
        auth_value = int(item[2])


        if not (0 <= index_value < 16):
            raise ValueError(
                f"Invalid fragment index: {index_value}"
            )

        if not (0 <= source_value <= 255):
            raise ValueError(
                f"Invalid source byte: {source_value}"
            )

        if not (0 <= auth_value <= 255):
            raise ValueError(
                f"Invalid auth byte: {auth_value}"
            )


        index_bits = [
            (index_value >> shift) & 1
            for shift in [3, 2, 1, 0]
        ]

        source_bits = [
            (source_value >> shift) & 1
            for shift in range(7, -1, -1)
        ]

        auth_bits = [
            (auth_value >> shift) & 1
            for shift in range(7, -1, -1)
        ]


        rows.append(
            index_bits
            +
            source_bits
            +
            auth_bits
        )


    bits20 = torch.tensor(
        rows,
        dtype=torch.float32,
        device=DEVICE,
    )


    if tuple(bits20.shape) != (16, 20):
        raise RuntimeError(
            f"Expected bits20=(16,20), got {tuple(bits20.shape)}"
        )


    return bits20



# ==============================================================================================================
# frozen_fragment_features
# ==============================================================================================================

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



# ==============================================================================================================
# load_eval_image
# ==============================================================================================================

def load_eval_image(
    path,
    image_size=256,
):

    image = (
        Image.open(
            path
        )
        .convert(
            "RGB"
        )
        .resize(
            (
                image_size,
                image_size,
            ),
            Image.Resampling.BICUBIC,
        )
    )

    array = np.asarray(
        image
    ).copy()

    tensor = (
        torch.from_numpy(
            array
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        /
        255.0
    )

    return tensor



# ==============================================================================================================
# make_physical_crop
# ==============================================================================================================

def make_physical_crop(
    image,
    crop_type,
    retain,
    rng,
):
    """
    image:
        [1,3,256,256]

    returns:
        crop:
            [1,3,h,w]

        metadata:
            dict with x/y/w/h
    """

    if image.shape != (1, 3, 256, 256):
        raise ValueError(
            f"Expected [1,3,256,256], got {tuple(image.shape)}"
        )


    # ----------------------------------------------------------------------------------------------
    # RANDOM SQUARE
    # ----------------------------------------------------------------------------------------------

    if crop_type == "random_square":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        max_xy = (
            256
            -
            side
        )

        x = rng.randint(
            0,
            max_xy,
        )

        y = rng.randint(
            0,
            max_xy,
        )

        w = side
        h = side


    # ----------------------------------------------------------------------------------------------
    # CENTER
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "center":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        w = side
        h = side

        x = (
            256
            -
            w
        ) // 2

        y = (
            256
            -
            h
        ) // 2


    # ----------------------------------------------------------------------------------------------
    # CORNER
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "corner":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        w = side
        h = side

        corner = rng.choice(
            [
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ]
        )

        if corner == "top_left":

            x = 0
            y = 0

        elif corner == "top_right":

            x = 256 - w
            y = 0

        elif corner == "bottom_left":

            x = 0
            y = 256 - h

        else:

            x = 256 - w
            y = 256 - h


    # ----------------------------------------------------------------------------------------------
    # EDGE
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "edge":

        side = int(
            round(
                256 * retain
            )
        )

        side = max(
            16,
            min(
                256,
                side,
            ),
        )

        w = side
        h = side

        edge = rng.choice(
            [
                "top",
                "bottom",
                "left",
                "right",
            ]
        )

        if edge == "top":

            y = 0

            x = rng.randint(
                0,
                256 - w,
            )

        elif edge == "bottom":

            y = 256 - h

            x = rng.randint(
                0,
                256 - w,
            )

        elif edge == "left":

            x = 0

            y = rng.randint(
                0,
                256 - h,
            )

        else:

            x = 256 - w

            y = rng.randint(
                0,
                256 - h,
            )


    # ----------------------------------------------------------------------------------------------
    # RANDOM RECTANGLE
    #
    # Original benchmark formulation:
    #
    # target area = 256^2 * retain^2
    # aspect ratio sampled log-uniformly 0.60 - 1.67
    # ----------------------------------------------------------------------------------------------

    elif crop_type == "random_rectangle":

        target_area = (
            256
            *
            256
            *
            retain
            *
            retain
        )

        ratio = math.exp(
            rng.uniform(
                math.log(
                    0.60
                ),
                math.log(
                    1.67
                ),
            )
        )

        w = int(
            round(
                math.sqrt(
                    target_area
                    *
                    ratio
                )
            )
        )

        h = int(
            round(
                math.sqrt(
                    target_area
                    /
                    ratio
                )
            )
        )

        w = max(
            32,
            min(
                256,
                w,
            ),
        )

        h = max(
            32,
            min(
                256,
                h,
            ),
        )

        x = rng.randint(
            0,
            256 - w,
        )

        y = rng.randint(
            0,
            256 - h,
        )


    else:

        raise ValueError(
            f"Unknown crop type: {crop_type}"
        )


    crop = image[
        :,
        :,
        y:y+h,
        x:x+w,
    ]


    metadata = {
        "crop_type":
            crop_type,

        "retain":
            float(
                retain
            ),

        "x":
            int(
                x
            ),

        "y":
            int(
                y
            ),

        "w":
            int(
                w
            ),

        "h":
            int(
                h
            ),
    }


    return (
        crop,
        metadata,
    )



# ==============================================================================================================
# normalize_target_bits
# ==============================================================================================================

def normalize_target_bits(
    target_bits20,
):

    if tuple(target_bits20.shape) == (
        4,
        4,
        20,
    ):

        target_bits20 = target_bits20.reshape(
            16,
            20,
        )

    elif tuple(target_bits20.shape) == (
        1,
        4,
        4,
        20,
    ):

        target_bits20 = target_bits20.reshape(
            16,
            20,
        )

    elif tuple(target_bits20.shape) == (
        1,
        16,
        20,
    ):

        target_bits20 = target_bits20[0]

    elif tuple(target_bits20.shape) == (
        16,
        20,
    ):

        pass

    else:

        raise RuntimeError(
            "Unexpected target_bits20 shape: "
            + str(
                tuple(
                    target_bits20.shape
                )
            )
        )


    return (
        target_bits20
        .to(DEVICE)
        .float()
    )



# ==============================================================================================================
# recover_irregular_phase
# ==============================================================================================================

def recover_irregular_phase(
    observations,
    key,
):

    candidates = generate_irregular_candidates(
        observations
    )


    verified = []


    for candidate_source, info in candidates.items():

        verification = verify_candidate_strict(
            candidate_source,
            observations,
            key,
        )


        if (
            verification[
                "valid_count"
            ]
            >=
            8
        ):

            verified.append(
                {
                    "source":
                        candidate_source,

                    "valid_count":
                        verification[
                            "valid_count"
                        ],

                    "supporting_subsets":
                        info[
                            "supporting_subsets"
                        ],
                }
            )


    verified.sort(
        key=lambda item:
            (
                item[
                    "valid_count"
                ],
                item[
                    "supporting_subsets"
                ],
            ),
        reverse=True,
    )


    if len(
        verified
    ) == 0:

        return {
            "status":
                "unresolved",

            "source":
                None,

            "valid_count":
                0,
        }


    sources = {
        item[
            "source"
        ]
        for item in verified
    }


    if len(
        sources
    ) > 1:

        return {
            "status":
                "ambiguous",

            "source":
                None,

            "valid_count":
                verified[
                    0
                ][
                    "valid_count"
                ],
        }


    return {
        "status":
            "accepted",

        "source":
            verified[
                0
            ][
                "source"
            ],

        "valid_count":
            verified[
                0
            ][
                "valid_count"
            ],
    }



# ==============================================================================================================
# recover_v410_at_placement
# ==============================================================================================================

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



# ==============================================================================================================
# recover_v413_at_placement
# ==============================================================================================================

@torch.no_grad()
def recover_v413_at_placement(
    crop,
    candidate_x,
    candidate_y,
):

    # ==============================================================================================
    # STAGE 1 — CURRENT VALIDATED V4.10
    # ==============================================================================================

    original = (
        _V413_ORIGINAL_RECOVER_AT_PLACEMENT(

            crop=crop,

            candidate_x=
                candidate_x,

            candidate_y=
                candidate_y,
        )
    )


    if original.get(
        "accepted",
        False,
    ):

        result = dict(
            original
        )


        result[
            "v413_method"
        ] = "original_top6"


        result[
            "subset_fallback_used"
        ] = False


        return result


    # ==============================================================================================
    # STAGE 2 — 6-OF-8 FALLBACK
    # ==============================================================================================

    subset = (
        recover_v413_subset_fallback_at_placement(

            crop=crop,

            candidate_x=
                candidate_x,

            candidate_y=
                candidate_y,

            pool_size=8,
        )
    )


    if subset.get(
        "accepted",
        False,
    ):

        result = dict(
            subset
        )


        result[
            "subset_fallback_used"
        ] = True


        result[
            "original_status"
        ] = original.get(
            "status"
        )


        return result


    # ==============================================================================================
    # PRESERVE AMBIGUITY
    # ==============================================================================================

    if subset.get(
        "status"
    ) == "ambiguous":

        result = dict(
            subset
        )


        result[
            "subset_fallback_used"
        ] = True


        result[
            "original_status"
        ] = original.get(
            "status"
        )


        return result


    # ==============================================================================================
    # BOTH FAILED
    # ==============================================================================================

    result = dict(
        original
    )


    result[
        "v413_method"
    ] = "unresolved"


    result[
        "subset_fallback_used"
    ] = True


    result[
        "subset_status"
    ] = subset.get(
        "status"
    )


    result[
        "tested_subsets"
    ] = subset.get(
        "tested_subsets",
        0,
    )


    return result



# ==============================================================================================================
# recover_v413_subset_fallback_at_placement
# ==============================================================================================================

@torch.no_grad()
def recover_v413_subset_fallback_at_placement(
    crop,
    candidate_x,
    candidate_y,
    pool_size=8,
):

    candidate_x = int(
        candidate_x
    )

    candidate_y = int(
        candidate_y
    )


    # ==============================================================================================
    # CANONICAL PLACEMENT
    # ==============================================================================================

    canvas = place_crop_on_canvas(

        crop=crop,

        candidate_x=
            candidate_x,

        candidate_y=
            candidate_y,

        canonical_size=256,

        neutral_value=0.5,
    )


    regions = (
        split_batch_16_regions_v2(
            canvas
        )
        .to(
            DEVICE
        )
    )


    # ==============================================================================================
    # VISIBILITY
    # ==============================================================================================

    visibility = region_visibility_weights(

        crop_height=int(
            crop.shape[-2]
        ),

        crop_width=int(
            crop.shape[-1]
        ),

        candidate_x=
            candidate_x,

        candidate_y=
            candidate_y,

        device=DEVICE,
    )


    if not torch.is_tensor(
        visibility
    ):

        visibility = torch.as_tensor(

            visibility,

            dtype=
                torch.float32,

            device=
                DEVICE,
        )


    visibility = (
        visibility
        .reshape(16)
        .float()
        .to(
            DEVICE
        )
    )


    # ==============================================================================================
    # EXACT SAME FEATURE PATH AS CURRENT V4.10
    # ==============================================================================================

    decoder = frozen_fragment_features(

        regions,

        visibility,
    )


    final_logits20 = decoder[
        "final"
    ]

    features = decoder[
        "features"
    ]


    # ==============================================================================================
    # SOURCE / AUTH BYTE SCORES
    # ==============================================================================================

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


    # ==============================================================================================
    # HARD TOP-1 BYTE PREDICTIONS
    #
    # Same byte decision rule as current V4.10.
    # ==============================================================================================

    source_bytes = (
        source_scores
        .argmax(
            dim=1
        )
        .detach()
        .cpu()
        .tolist()
    )


    auth_bytes = (
        auth_scores
        .argmax(
            dim=1
        )
        .detach()
        .cpu()
        .tolist()
    )


    # ==============================================================================================
    # CONFIDENCE
    #
    # Used ONLY to rank authenticated solutions.
    # Never used to bypass HMAC.
    # ==============================================================================================

    source_conf = (
        torch.softmax(
            source_scores,
            dim=1,
        )
        .max(
            dim=1
        )
        .values
    )


    auth_conf = (
        torch.softmax(
            auth_scores,
            dim=1,
        )
        .max(
            dim=1
        )
        .values
    )


    # ==============================================================================================
    # BEST VISIBLE REGION POOL
    # ==============================================================================================

    ranked_positions = (
        torch.argsort(
            visibility,
            descending=True,
        )
        .tolist()
    )


    visible_positions = [

        p

        for p in ranked_positions

        if float(
            visibility[
                p
            ].item()
        ) > 0.0

    ]


    pool = visible_positions[
        :pool_size
    ]


    # Need at least six visible RS symbols.
    if len(
        pool
    ) < 6:

        return {

            "accepted":
                False,

            "status":
                "insufficient_visible_regions",

            "token":
                None,

            "auth":
                None,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "pool_positions":
                tuple(
                    pool
                ),

            "tested_subsets":
                0,

            "visibility":
                visibility,

            "source_bytes":
                source_bytes,

            "auth_bytes":
                auth_bytes,

            "v413_method":
                "subset_6_of_8",
        }


    # ==============================================================================================
    # TRY EVERY SIX-REGION SUBSET INSIDE TOP-8
    #
    # C(8,6) = 28 when 8 visible regions exist.
    # ==============================================================================================

    authenticated = []

    tested_subsets = 0


    for subset in combinations(
        pool,
        6,
    ):

        tested_subsets += 1


        known_positions = list(
            subset
        )


        # ------------------------------------------------------------------------------------------
        # Construct observed source codeword
        # ------------------------------------------------------------------------------------------

        source_observed = [

            0

            for _ in range(16)

        ]


        auth_observed = [

            0

            for _ in range(16)

        ]


        for p in known_positions:

            source_observed[
                p
            ] = int(
                source_bytes[
                    p
                ]
            )


            auth_observed[
                p
            ] = int(
                auth_bytes[
                    p
                ]
            )


        # ------------------------------------------------------------------------------------------
        # RS(16,6)
        # ------------------------------------------------------------------------------------------

        recovered_token = (
            v4_try_rs_configuration(

                observed_bytes=
                    source_observed,

                known_positions=
                    known_positions,
            )
        )


        if recovered_token is None:

            continue


        recovered_auth = (
            v4_try_rs_configuration(

                observed_bytes=
                    auth_observed,

                known_positions=
                    known_positions,
            )
        )


        if recovered_auth is None:

            continue


        recovered_token = bytes(
            recovered_token
        )


        recovered_auth = bytes(
            recovered_auth
        )


        # ------------------------------------------------------------------------------------------
        # EXACT HMAC — SECURITY CONDITION UNCHANGED
        # ------------------------------------------------------------------------------------------

        expected_auth = v4_compute_auth_tag(

            recovered_token,

            V3H_TRAIN_KEY,
        )


        if not hmac.compare_digest(
            recovered_auth,
            expected_auth,
        ):

            continue


        # ------------------------------------------------------------------------------------------
        # Authenticated subset support
        # ------------------------------------------------------------------------------------------

        support = 0.0


        for p in known_positions:

            support += (

                float(
                    visibility[
                        p
                    ].item()
                )

                *

                0.5

                *

                (
                    float(
                        source_conf[
                            p
                        ].item()
                    )

                    +

                    float(
                        auth_conf[
                            p
                        ].item()
                    )
                )
            )


        authenticated.append(

            {

                "token":
                    recovered_token,

                "auth":
                    recovered_auth,

                "positions":
                    tuple(
                        known_positions
                    ),

                "support":
                    float(
                        support
                    ),
            }

        )


    # ==============================================================================================
    # NO SUBSET AUTHENTICATED
    # ==============================================================================================

    if len(
        authenticated
    ) == 0:

        return {

            "accepted":
                False,

            "status":
                "subset_auth_failed",

            "token":
                None,

            "auth":
                None,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "pool_positions":
                tuple(
                    pool
                ),

            "tested_subsets":
                tested_subsets,

            "visibility":
                visibility,

            "source_bytes":
                source_bytes,

            "auth_bytes":
                auth_bytes,

            "v413_method":
                "subset_6_of_8",
        }


    # ==============================================================================================
    # DEDUPLICATE AUTHENTICATED IDENTITIES
    # ==============================================================================================

    best_by_token = {}


    for item in authenticated:

        token = item[
            "token"
        ]


        previous = best_by_token.get(
            token
        )


        if (
            previous is None
            or
            item[
                "support"
            ]
            >
            previous[
                "support"
            ]
        ):

            best_by_token[
                token
            ] = item


    unique = list(
        best_by_token.values()
    )


    unique.sort(

        key=lambda item:
            item[
                "support"
            ],

        reverse=True,
    )


    # ==============================================================================================
    # MULTIPLE DISTINCT AUTHENTICATED TOKENS = AMBIGUOUS
    # ==============================================================================================

    if len(
        unique
    ) > 1:

        return {

            "accepted":
                False,

            "status":
                "ambiguous",

            "ambiguous":
                True,

            "token":
                None,

            "auth":
                None,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "pool_positions":
                tuple(
                    pool
                ),

            "tested_subsets":
                tested_subsets,

            "authenticated_candidates":
                unique,

            "visibility":
                visibility,

            "source_bytes":
                source_bytes,

            "auth_bytes":
                auth_bytes,

            "v413_method":
                "subset_6_of_8",
        }


    # ==============================================================================================
    # UNIQUE AUTHENTICATED TOKEN
    # ==============================================================================================

    best = unique[
        0
    ]


    return {

        "accepted":
            True,

        "status":
            "accepted",

        "ambiguous":
            False,

        "token":
            best[
                "token"
            ],

        "auth":
            best[
                "auth"
            ],

        "x":
            candidate_x,

        "y":
            candidate_y,

        "known_positions":
            best[
                "positions"
            ],

        "pool_positions":
            tuple(
                pool
            ),

        "tested_subsets":
            tested_subsets,

        "support":
            float(
                best[
                    "support"
                ]
            ),

        "visibility":
            visibility,

        "source_bytes":
            source_bytes,

        "auth_bytes":
            auth_bytes,

        "v413_method":
            "subset_6_of_8",
    }



# ==============================================================================================================
# recover_v414_at_placement
# ==============================================================================================================

@torch.no_grad()
def recover_v414_at_placement(
    crop,
    candidate_x,
    candidate_y,
):

    existing = _V414_V413_RECOVER(

        crop=crop,

        candidate_x=
            candidate_x,

        candidate_y=
            candidate_y,
    )


    if existing.get(
        "accepted",
        False,
    ):

        result = dict(
            existing
        )

        result[
            "v414_method"
        ] = result.get(
            "v413_method",
            "existing",
        )

        return result


    if existing.get(
        "status"
    ) == "ambiguous":

        return existing


    decoupled = (
        recover_v414_decoupled_at_placement(

            crop=crop,

            candidate_x=
                candidate_x,

            candidate_y=
                candidate_y,

            pool_size=8,
        )
    )


    if decoupled.get(
        "accepted",
        False,
    ):

        decoupled[
            "previous_status"
        ] = existing.get(
            "status"
        )

        return decoupled


    if decoupled.get(
        "status"
    ) == "ambiguous":

        return decoupled


    result = dict(
        existing
    )

    result[
        "v414_method"
    ] = "unresolved"

    result[
        "decoupled_status"
    ] = decoupled.get(
        "status"
    )

    result[
        "source_candidate_count"
    ] = decoupled.get(
        "source_candidate_count",
        0,
    )

    result[
        "auth_candidate_count"
    ] = decoupled.get(
        "auth_candidate_count",
        0,
    )

    return result



# ==============================================================================================================
# recover_v414_decoupled_at_placement
# ==============================================================================================================

@torch.no_grad()
def recover_v414_decoupled_at_placement(
    crop,
    candidate_x,
    candidate_y,
    pool_size=8,
):

    candidate_x = int(
        candidate_x
    )

    candidate_y = int(
        candidate_y
    )


    # ==============================================================================================
    # CANVAS / VISIBILITY
    # ==============================================================================================

    canvas = place_crop_on_canvas(

        crop=crop,

        candidate_x=
            candidate_x,

        candidate_y=
            candidate_y,

        canonical_size=256,

        neutral_value=0.5,
    )


    regions = (
        split_batch_16_regions_v2(
            canvas
        )
        .to(
            DEVICE
        )
    )


    visibility = region_visibility_weights(

        crop_height=int(
            crop.shape[-2]
        ),

        crop_width=int(
            crop.shape[-1]
        ),

        candidate_x=
            candidate_x,

        candidate_y=
            candidate_y,

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
        .to(
            DEVICE
        )
    )


    # ==============================================================================================
    # SAME TRAINED DECODER
    # ==============================================================================================

    decoder = frozen_fragment_features(

        regions,

        visibility,
    )


    final_logits20 = decoder[
        "final"
    ]

    features = decoder[
        "features"
    ]


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


    source_bytes = (
        source_scores
        .argmax(
            dim=1
        )
        .detach()
        .cpu()
        .tolist()
    )


    auth_bytes = (
        auth_scores
        .argmax(
            dim=1
        )
        .detach()
        .cpu()
        .tolist()
    )


    # ==============================================================================================
    # REGION POOL
    # ==============================================================================================

    ranked_positions = (
        torch.argsort(
            visibility,
            descending=True,
        )
        .tolist()
    )


    visible_positions = [

        p

        for p in ranked_positions

        if float(
            visibility[
                p
            ].item()
        ) > 0.0

    ]


    pool = visible_positions[
        :pool_size
    ]


    if len(
        pool
    ) < 6:

        return {

            "accepted":
                False,

            "status":
                "insufficient_visible_regions",

            "token":
                None,

            "auth":
                None,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "v414_method":
                "decoupled_6_of_8",
        }


    # ==============================================================================================
    # SOURCE CANDIDATES
    # ==============================================================================================

    source_candidates = {}


    for subset in combinations(
        pool,
        6,
    ):

        known_positions = list(
            subset
        )


        observed = [
            0
            for _ in range(16)
        ]


        for p in known_positions:

            observed[
                p
            ] = int(
                source_bytes[
                    p
                ]
            )


        recovered = v4_try_rs_configuration(

            observed_bytes=
                observed,

            known_positions=
                known_positions,
        )


        if recovered is None:

            continue


        recovered = bytes(
            recovered
        )


        if recovered not in source_candidates:

            source_candidates[
                recovered
            ] = []


        source_candidates[
            recovered
        ].append(
            tuple(
                known_positions
            )
        )


    # ==============================================================================================
    # AUTH CANDIDATES
    # ==============================================================================================

    auth_candidates = {}


    for subset in combinations(
        pool,
        6,
    ):

        known_positions = list(
            subset
        )


        observed = [
            0
            for _ in range(16)
        ]


        for p in known_positions:

            observed[
                p
            ] = int(
                auth_bytes[
                    p
                ]
            )


        recovered = v4_try_rs_configuration(

            observed_bytes=
                observed,

            known_positions=
                known_positions,
        )


        if recovered is None:

            continue


        recovered = bytes(
            recovered
        )


        if recovered not in auth_candidates:

            auth_candidates[
                recovered
            ] = []


        auth_candidates[
            recovered
        ].append(
            tuple(
                known_positions
            )
        )


    # ==============================================================================================
    # EXACT CRYPTOGRAPHIC JOIN
    #
    # We do NOT choose the closest auth candidate.
    #
    # We compute the exact expected HMAC for every source candidate
    # and require an exact match among independently reconstructed
    # auth candidates.
    # ==============================================================================================

    authenticated = []


    for token, source_subsets in (
        source_candidates.items()
    ):

        expected_auth = v4_compute_auth_tag(

            token,

            V3H_TRAIN_KEY,
        )


        if expected_auth not in auth_candidates:

            continue


        authenticated.append(

            {

                "token":
                    token,

                "auth":
                    expected_auth,

                "source_subsets":
                    source_subsets,

                "auth_subsets":
                    auth_candidates[
                        expected_auth
                    ],

                "source_support_count":
                    len(
                        source_subsets
                    ),

                "auth_support_count":
                    len(
                        auth_candidates[
                            expected_auth
                        ]
                    ),
            }

        )


    # ==============================================================================================
    # RESULT
    # ==============================================================================================

    if len(
        authenticated
    ) == 0:

        return {

            "accepted":
                False,

            "status":
                "decoupled_auth_failed",

            "token":
                None,

            "auth":
                None,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "source_candidate_count":
                len(
                    source_candidates
                ),

            "auth_candidate_count":
                len(
                    auth_candidates
                ),

            "pool_positions":
                tuple(
                    pool
                ),

            "v414_method":
                "decoupled_6_of_8",
        }


    unique_tokens = {

        item[
            "token"
        ]

        for item in authenticated

    }


    if len(
        unique_tokens
    ) > 1:

        return {

            "accepted":
                False,

            "status":
                "ambiguous",

            "ambiguous":
                True,

            "token":
                None,

            "auth":
                None,

            "authenticated_candidates":
                authenticated,

            "x":
                candidate_x,

            "y":
                candidate_y,

            "v414_method":
                "decoupled_6_of_8",
        }


    # strongest support first
    authenticated.sort(

        key=lambda item: (

            item[
                "source_support_count"
            ]

            +

            item[
                "auth_support_count"
            ]

        ),

        reverse=True,
    )


    best = authenticated[
        0
    ]


    return {

        "accepted":
            True,

        "status":
            "accepted",

        "ambiguous":
            False,

        "token":
            best[
                "token"
            ],

        "auth":
            best[
                "auth"
            ],

        "x":
            candidate_x,

        "y":
            candidate_y,

        "source_subsets":
            best[
                "source_subsets"
            ],

        "auth_subsets":
            best[
                "auth_subsets"
            ],

        "source_candidate_count":
            len(
                source_candidates
            ),

        "auth_candidate_count":
            len(
                auth_candidates
            ),

        "pool_positions":
            tuple(
                pool
            ),

        "v414_method":
            "decoupled_6_of_8",
    }



# ==============================================================================================================
# recover_v415_at_placement
# ==============================================================================================================

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



# ==============================================================================================================
# recover_v415_soft_auth_at_placement
# ==============================================================================================================

@torch.no_grad()
def recover_v415_soft_auth_at_placement(
    crop,
    candidate_x,
    candidate_y,
    source_pool_size=8,
    auth_pool_size=8,
    auth_top_k=4,
    min_auth_support=6,
):

    candidate_x = int(candidate_x)
    candidate_y = int(candidate_y)

    # ----------------------------------------------------------------------------------------------
    # Reconstruct candidate canvas
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
    ).to(DEVICE)

    # ----------------------------------------------------------------------------------------------
    # Visibility
    # ----------------------------------------------------------------------------------------------

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
    # Same frozen decoder path
    # ----------------------------------------------------------------------------------------------

    decoder = frozen_fragment_features(
        regions,
        visibility,
    )

    final_logits20 = decoder["final"]
    features = decoder["features"]

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
    # Hard source predictions
    # ----------------------------------------------------------------------------------------------

    source_bytes = (
        source_scores
        .argmax(dim=1)
        .detach()
        .cpu()
        .tolist()
    )

    # ----------------------------------------------------------------------------------------------
    # Soft auth Top-k candidates
    # ----------------------------------------------------------------------------------------------

    auth_topk = (
        torch.topk(
            auth_scores,
            k=auth_top_k,
            dim=1,
        )
        .indices
        .detach()
        .cpu()
        .tolist()
    )

    auth_probs = torch.softmax(
        auth_scores,
        dim=1,
    )

    # ----------------------------------------------------------------------------------------------
    # Visible-region ordering
    # ----------------------------------------------------------------------------------------------

    ranked_positions = (
        torch.argsort(
            visibility,
            descending=True,
        )
        .tolist()
    )

    visible_positions = [
        p
        for p in ranked_positions
        if float(visibility[p].item()) > 0.0
    ]

    source_pool = visible_positions[:source_pool_size]
    auth_pool = visible_positions[:auth_pool_size]

    if len(source_pool) < 6:
        return {
            "accepted": False,
            "status": "insufficient_visible_regions",
            "token": None,
            "auth": None,
            "x": candidate_x,
            "y": candidate_y,
            "v415_method": "soft_auth",
        }

    # ----------------------------------------------------------------------------------------------
    # Generate possible source tokens from all 6-of-8 source subsets
    # ----------------------------------------------------------------------------------------------

    source_candidates = {}

    for subset in combinations(
        source_pool,
        6,
    ):

        known_positions = list(subset)

        observed = [0] * 16

        for p in known_positions:
            observed[p] = int(
                source_bytes[p]
            )

        recovered = v4_try_rs_configuration(
            observed_bytes=observed,
            known_positions=known_positions,
        )

        if recovered is None:
            continue

        token = bytes(recovered)

        source_candidates.setdefault(
            token,
            []
        ).append(
            tuple(known_positions)
        )

    if not source_candidates:
        return {
            "accepted": False,
            "status": "source_rs_failed",
            "token": None,
            "auth": None,
            "x": candidate_x,
            "y": candidate_y,
            "v415_method": "soft_auth",
        }

    # ----------------------------------------------------------------------------------------------
    # Exact HMAC + exact RS codeword check against soft auth predictions
    # ----------------------------------------------------------------------------------------------

    authenticated = []

    for token, source_subsets in source_candidates.items():

        expected_auth_tag = bytes(
            v4_compute_auth_tag(
                token,
                V3H_TRAIN_KEY,
            )
        )

        expected_auth_codeword = bytes(
            V4_RS_CODEC.encode(
                expected_auth_tag
            )
        )

        if len(expected_auth_codeword) != 16:
            raise RuntimeError(
                f"Expected 16-byte auth codeword, got {len(expected_auth_codeword)}"
            )

        supported_positions = []
        support_score = 0.0

        for p in auth_pool:

            expected_byte = int(
                expected_auth_codeword[p]
            )

            candidates = auth_topk[p]

            if expected_byte not in candidates:
                continue

            rank = candidates.index(
                expected_byte
            ) + 1

            prob = float(
                auth_probs[
                    p,
                    expected_byte
                ].item()
            )

            supported_positions.append(
                (
                    int(p),
                    int(rank),
                    prob,
                )
            )

            support_score += (
                float(
                    visibility[p].item()
                )
                *
                prob
            )

        if len(supported_positions) < min_auth_support:
            continue

        authenticated.append(
            {
                "token": token,
                "auth": expected_auth_tag,
                "source_subsets": source_subsets,
                "auth_supported_positions": supported_positions,
                "auth_support_count": len(supported_positions),
                "support": float(support_score),
            }
        )

    # ----------------------------------------------------------------------------------------------
    # No valid authenticated candidate
    # ----------------------------------------------------------------------------------------------

    if not authenticated:

        return {
            "accepted": False,
            "status": "soft_auth_failed",
            "token": None,
            "auth": None,
            "x": candidate_x,
            "y": candidate_y,
            "source_candidate_count": len(source_candidates),
            "v415_method": "soft_auth",
        }

    # ----------------------------------------------------------------------------------------------
    # Deduplicate tokens
    # ----------------------------------------------------------------------------------------------

    best_by_token = {}

    for item in authenticated:

        token = item["token"]

        previous = best_by_token.get(
            token
        )

        if (
            previous is None
            or
            item["support"] > previous["support"]
        ):
            best_by_token[token] = item

    unique = list(
        best_by_token.values()
    )

    unique.sort(
        key=lambda item: (
            item["auth_support_count"],
            item["support"],
            len(item["source_subsets"]),
        ),
        reverse=True,
    )

    # Multiple independently authenticated tokens -> ambiguous
    if len(unique) > 1:

        return {
            "accepted": False,
            "status": "ambiguous",
            "ambiguous": True,
            "token": None,
            "auth": None,
            "authenticated_candidates": unique,
            "x": candidate_x,
            "y": candidate_y,
            "v415_method": "soft_auth",
        }

    best = unique[0]

    return {
        "accepted": True,
        "status": "accepted",
        "ambiguous": False,
        "token": best["token"],
        "auth": best["auth"],
        "x": candidate_x,
        "y": candidate_y,
        "source_subsets": best["source_subsets"],
        "auth_supported_positions": best["auth_supported_positions"],
        "auth_support_count": best["auth_support_count"],
        "support": best["support"],
        "source_candidate_count": len(source_candidates),
        "v415_method": "soft_auth",
    }



# ==============================================================================================================
# score_spatial_index_batch
# ==============================================================================================================

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



# ==============================================================================================================
# source_byte_scores_local
# ==============================================================================================================

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



# ==============================================================================================================
# split_16_regions
# ==============================================================================================================

def split_16_regions(
    image,
):

    if tuple(
        image.shape[
            1:
        ]
    ) != (
        3,
        256,
        256,
    ):
        raise ValueError(
            "Expected [B,3,256,256]"
        )


    tiles = []


    for row in range(
        4
    ):

        for col in range(
            4
        ):

            y0 = row * 64
            x0 = col * 64

            tiles.append(
                image[
                    :,
                    :,
                    y0:y0 + 64,
                    x0:x0 + 64,
                ]
            )


    return torch.cat(
        tiles,
        dim=0,
    )



# ==============================================================================================================
# v4_compute_auth_tag
# ==============================================================================================================

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



# ==============================================================================================================
# v4_issue
# ==============================================================================================================

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



# ==============================================================================================================
# v4_rs_decode
# ==============================================================================================================

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



# ==============================================================================================================
# v4_source_token
# ==============================================================================================================

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



# ==============================================================================================================
# v4_try_rs_configuration
# ==============================================================================================================

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

