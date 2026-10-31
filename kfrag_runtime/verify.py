import torch

from .runtime_ops import blind_crop_search_spatial_fast
from .recovery import recover_v415_at_placement


@torch.no_grad()
def verify_crop_v415(
    crop,
    runtime,
    auth_key,
    auth_domain,
    top_k=20,
):
    """
    Full K-FRAG V4.15 crop verification.

    Pipeline:
        crop
        -> blind spatial synchronization
        -> ranked candidate placements
        -> V4.15 authenticated recovery

    Returns a dictionary describing the final result.
    """

    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a dictionary.")

    if not isinstance(auth_key, (bytes, bytearray)):
        raise TypeError("auth_key must be bytes or bytearray.")

    if not isinstance(auth_domain, (bytes, bytearray)):
        raise TypeError("auth_domain must be bytes or bytearray.")

    if crop.ndim == 4:
        if crop.shape[0] != 1:
            raise ValueError(
                f"Expected one crop, got shape {tuple(crop.shape)}"
            )

    elif crop.ndim != 3:
        raise ValueError(
            f"Expected [3,H,W] or [1,3,H,W], got {tuple(crop.shape)}"
        )

    # -----------------------------------------------------------------
    # Stage 1: blind spatial synchronization
    # -----------------------------------------------------------------

    spatial = blind_crop_search_spatial_fast(
        crop,
        runtime,
    )

    ranked = spatial.get("ranked", [])

    if not ranked:
        return {
            "accepted": False,
            "status": "no_spatial_candidates",
            "recovery": None,
            "spatial": spatial,
        }

    candidate_limit = min(
        int(top_k),
        len(ranked),
    )

    ambiguous_results = []
    attempted = []

    # -----------------------------------------------------------------
    # Stage 2: authenticated recovery on ranked placements
    # -----------------------------------------------------------------

    for rank, candidate in enumerate(
        ranked[:candidate_limit],
        start=1,
    ):
        x = int(candidate["x"])
        y = int(candidate["y"])

        recovery = recover_v415_at_placement(
            crop=crop,
            candidate_x=x,
            candidate_y=y,
            runtime=runtime,
            auth_key=auth_key,
            auth_domain=auth_domain,
        )

        attempt = {
            "rank": rank,
            "x": x,
            "y": y,
            "spatial_score": float(
                candidate.get("score", 0.0)
            ),
            "hard_alignment": float(
                candidate.get("hard_alignment", 0.0)
            ),
            "confidence": float(
                candidate.get("confidence", 0.0)
            ),
            "recovery_status": recovery.get("status"),
            "accepted": bool(
                recovery.get("accepted", False)
            ),
        }

        attempted.append(attempt)

        # -------------------------------------------------------------
        # Exact authenticated success
        # -------------------------------------------------------------

        if recovery.get("accepted", False):
            return {
                "accepted": True,
                "status": "accepted",
                "rank": rank,
                "x": x,
                "y": y,
                "candidate": candidate,
                "recovery": recovery,
                "attempted": attempted,
                "spatial": spatial,
            }

        # -------------------------------------------------------------
        # Preserve ambiguity instead of treating it as ordinary failure
        # -------------------------------------------------------------

        if recovery.get("status") == "ambiguous":
            ambiguous_results.append(
                {
                    "rank": rank,
                    "candidate": candidate,
                    "recovery": recovery,
                }
            )

    # -----------------------------------------------------------------
    # Stage 3: no authenticated placement
    # -----------------------------------------------------------------

    if ambiguous_results:
        return {
            "accepted": False,
            "status": "ambiguous",
            "ambiguous_results": ambiguous_results,
            "attempted": attempted,
            "spatial": spatial,
        }

    return {
        "accepted": False,
        "status": "authentication_failed",
        "recovery": None,
        "attempted": attempted,
        "spatial": spatial,
    }


__all__ = [
    "verify_crop_v415",
]