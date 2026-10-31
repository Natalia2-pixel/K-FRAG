"""
K-FRAG V4.15 recovery stack.

The recovery algorithms in this module originate from the recovered frozen
V4.10 -> V4.15 implementation.

The public API receives the runtime, authentication key and authentication
domain explicitly. Private authentication material is never stored in this
repository.

The internal compatibility context exists only to execute the recovered
function bodies without changing their decision logic.
"""

from itertools import combinations
import hashlib
import hmac
from threading import RLock

import torch

from .runtime_ops import (
    frozen_fragment_features as _runtime_frozen_fragment_features,
    place_crop_on_canvas as _runtime_place_crop_on_canvas,
    region_visibility_weights,
    split_batch_16_regions_v2,
)

from .protocol_v4 import (
    V4_RS_CODEC,
    auth_byte_scores_local,
    source_byte_scores_local,
    v4_compute_auth_tag as _protocol_v4_compute_auth_tag,
    v4_try_rs_configuration,
)


# ---------------------------------------------------------------------
# Compatibility context
# ---------------------------------------------------------------------

_CONTEXT_LOCK = RLock()

_ACTIVE_RUNTIME = None
_ACTIVE_AUTH_DOMAIN = None

DEVICE = None
weak_symbol_head = None
auth_symbol_head = None
V3H_TRAIN_KEY = None


def _validate_context(runtime, auth_key, auth_domain):
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a dictionary.")

    required = (
        "device",
        "weak_symbol_head",
        "auth_symbol_head",
    )

    missing = [
        name
        for name in required
        if name not in runtime
    ]

    if missing:
        raise KeyError(
            "Runtime is missing: "
            + ", ".join(missing)
        )

    if not isinstance(auth_key, (bytes, bytearray)):
        raise TypeError(
            "auth_key must be bytes or bytearray."
        )

    if not isinstance(auth_domain, (bytes, bytearray)):
        raise TypeError(
            "auth_domain must be bytes or bytearray."
        )


def frozen_fragment_features(regions, visibility):
    """
    Compatibility adapter for the recovered frozen recovery functions.

    The clean runtime implementation receives runtime explicitly.
    """
    if _ACTIVE_RUNTIME is None:
        raise RuntimeError(
            "K-FRAG recovery runtime context is not active."
        )

    return _runtime_frozen_fragment_features(
        regions,
        visibility,
        _ACTIVE_RUNTIME,
    )


def v4_compute_auth_tag(token48, key):
    """
    Compatibility adapter preserving the recovered two-argument call.

    The public protocol implementation receives auth_domain explicitly.
    """
    if _ACTIVE_AUTH_DOMAIN is None:
        raise RuntimeError(
            "K-FRAG authentication domain is not active."
        )

    return _protocol_v4_compute_auth_tag(
        token48,
        key,
        _ACTIVE_AUTH_DOMAIN,
    )


def _invoke_recovered(
    function,
    *,
    runtime,
    auth_key,
    auth_domain,
    kwargs,
):
    """
    Execute one recovered recovery function with explicit public dependencies.

    The previous module state is restored afterwards so authentication
    material is not retained after the call.
    """
    global _ACTIVE_RUNTIME
    global _ACTIVE_AUTH_DOMAIN
    global DEVICE
    global weak_symbol_head
    global auth_symbol_head
    global V3H_TRAIN_KEY

    _validate_context(
        runtime,
        auth_key,
        auth_domain,
    )

    with _CONTEXT_LOCK:
        previous = (
            _ACTIVE_RUNTIME,
            _ACTIVE_AUTH_DOMAIN,
            DEVICE,
            weak_symbol_head,
            auth_symbol_head,
            V3H_TRAIN_KEY,
        )

        try:
            _ACTIVE_RUNTIME = runtime
            _ACTIVE_AUTH_DOMAIN = bytes(auth_domain)

            DEVICE = runtime["device"]
            weak_symbol_head = runtime["weak_symbol_head"]
            auth_symbol_head = runtime["auth_symbol_head"]

            # Exists only while this recovery call executes.
            V3H_TRAIN_KEY = bytes(auth_key)

            return function(**kwargs)

        finally:
            (
                _ACTIVE_RUNTIME,
                _ACTIVE_AUTH_DOMAIN,
                DEVICE,
                weak_symbol_head,
                auth_symbol_head,
                V3H_TRAIN_KEY,
            ) = previous

def place_crop_on_canvas(
    crop,
    candidate_x,
    candidate_y,
    canonical_size=256,
    neutral_value=0.5,
):
    canvas = _runtime_place_crop_on_canvas(
        crop=crop,
        candidate_x=candidate_x,
        candidate_y=candidate_y,
        canonical_size=canonical_size,
        neutral_value=neutral_value,
    )

    if canvas.ndim == 3:
        canvas = canvas.unsqueeze(0)

    return canvas
# =====================================================================
# EXACT RECOVERED FUNCTION BODIES BEGIN BELOW
# =====================================================================

# Exact extracted K-FRAG recovery functions
# TEMPORARY RECOVERY REFERENCE - DO NOT COMMIT

# ==============================================================================
# _legacy_recover_v410_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v410_at_placement(
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


# ==============================================================================
# _legacy_recover_v413_subset_fallback_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v413_subset_fallback_at_placement(
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


# ==============================================================================
# _legacy_recover_v413_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v413_at_placement(
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
        _legacy_recover_v413_subset_fallback_at_placement(

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


# ==============================================================================
# _legacy_recover_v414_decoupled_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v414_decoupled_at_placement(
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


# ==============================================================================
# _legacy_recover_v414_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v414_at_placement(
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
        _legacy_recover_v414_decoupled_at_placement(

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


# ==============================================================================
# _legacy_recover_v415_soft_auth_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v415_soft_auth_at_placement(
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


# ==============================================================================
# _legacy_recover_v415_at_placement
# ==============================================================================
@torch.no_grad()
def _legacy_recover_v415_at_placement(
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

    soft = _legacy_recover_v415_soft_auth_at_placement(
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



# =====================================================================
# Historical recovery lineage
# =====================================================================

_V413_ORIGINAL_RECOVER_AT_PLACEMENT = (
    _legacy_recover_v410_at_placement
)

_V414_V413_RECOVER = (
    _legacy_recover_v413_at_placement
)

_V415_V414_RECOVER = (
    _legacy_recover_v414_at_placement
)


# =====================================================================
# Clean public API
# =====================================================================


@torch.no_grad()
def recover_v410_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
):
    return _invoke_recovered(
        _legacy_recover_v410_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
        },
    )


@torch.no_grad()
def recover_v413_subset_fallback_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
):
    return _invoke_recovered(
        _legacy_recover_v413_subset_fallback_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
        },
    )


@torch.no_grad()
def recover_v413_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
):
    return _invoke_recovered(
        _legacy_recover_v413_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
        },
    )


@torch.no_grad()
def recover_v414_decoupled_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
    pool_size=8,
):
    return _invoke_recovered(
        _legacy_recover_v414_decoupled_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
            "pool_size": pool_size,
        },
    )


@torch.no_grad()
def recover_v414_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
):
    return _invoke_recovered(
        _legacy_recover_v414_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
        },
    )


@torch.no_grad()
def recover_v415_soft_auth_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
    source_pool_size=8,
    auth_pool_size=8,
    auth_top_k=4,
    min_auth_support=6,
):
    return _invoke_recovered(
        _legacy_recover_v415_soft_auth_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
            "source_pool_size": source_pool_size,
            "auth_pool_size": auth_pool_size,
            "auth_top_k": auth_top_k,
            "min_auth_support": min_auth_support,
        },
    )


@torch.no_grad()
def recover_v415_at_placement(
    crop,
    candidate_x,
    candidate_y,
    runtime,
    auth_key,
    auth_domain,
):
    """
    Public K-FRAG V4.15 recovery entry point.

    Authentication material is supplied externally and is never embedded in
    the repository.
    """
    return _invoke_recovered(
        _legacy_recover_v415_at_placement,
        runtime=runtime,
        auth_key=auth_key,
        auth_domain=auth_domain,
        kwargs={
            "crop": crop,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
        },
    )


__all__ = [
    "recover_v410_at_placement",
    "recover_v413_subset_fallback_at_placement",
    "recover_v413_at_placement",
    "recover_v414_decoupled_at_placement",
    "recover_v414_at_placement",
    "recover_v415_soft_auth_at_placement",
    "recover_v415_at_placement",
]
