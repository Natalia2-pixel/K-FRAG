
"""Compatibility adapter for the archived K-FRAG V4.15 recovery path.

Preserves the historical blind_v412_phase_recovery function body.
Does not modify the public runtime_ops search configuration.

The caller must provide a loaded runtime and a locally supplied
authentication key. This adapter is not thread-safe with other
concurrent recovery operations.
"""

from threading import RLock
from kfrag_runtime import runtime_ops as ops
from kfrag_runtime import frozen_runtime_functions as frozen
from kfrag_runtime.recovery import recover_v410_at_placement


_HISTORICAL_LOCK = RLock()

_HISTORICAL_SEARCH = {
    "COARSE_STRIDE": 2,
    "TOP_COARSE": 60,
    "REFINE_RADIUS": 10,
    "BATCH_PLACEMENTS": 24,
}


def recover_historical_v415(
    crop,
    *,
    runtime,
    auth_key,
    auth_domain,
):
    """Run historical blind recovery with explicit runtime context."""

    if runtime is None:
        raise ValueError("A loaded V4.15 runtime is required.")

    if auth_key is None:
        raise ValueError("An authentication key is required.")

    if auth_domain is None:
        raise ValueError("An authentication domain is required.")

    historical_globals = (
        frozen.blind_v412_phase_recovery.__wrapped__.__globals__
        if hasattr(frozen.blind_v412_phase_recovery, "__wrapped__")
        else frozen.blind_v412_phase_recovery.__globals__
    )

    search_function = frozen.blind_crop_search_spatial_fast
    search_globals = search_function.__globals__

    def authenticated_placement(
        crop,
        candidate_x,
        candidate_y,
    ):
        return recover_v410_at_placement(
            crop=crop,
            candidate_x=candidate_x,
            candidate_y=candidate_y,
            runtime=runtime,
            auth_key=auth_key,
            auth_domain=auth_domain,
        )

    def historical_batch_canvases(
        crop,
        placements,
        canonical_size=256,
        neutral_value=0.5,
    ):
        return ops._batch_candidate_canvases(
            crop=crop,
            placements=placements,
            runtime=runtime,
            canonical_size=canonical_size,
            neutral_value=neutral_value,
        )

    def historical_decode_tiles(tiles):
        return ops.decode_phase_tiles_v2(
            tiles,
            runtime=runtime,
        )

    changes = [
        (
            historical_globals,
            "recover_v410_at_placement",
            authenticated_placement,
        ),
        (
            search_globals,
            "_batch_candidate_canvases",
            historical_batch_canvases,
        ),
        (
            search_globals,
            "decode_phase_tiles_v2",
            historical_decode_tiles,
        ),
        (
            search_globals,
            "canonical_index_bits",
            ops.canonical_index_bits,
        ),
    ]

    changes.extend(
        (search_globals, name, value)
        for name, value in _HISTORICAL_SEARCH.items()
    )

    with _HISTORICAL_LOCK:
        original_values = []

        try:
            for namespace, name, value in changes:
                existed = name in namespace
                previous = namespace.get(name)

                original_values.append(
                    (namespace, name, existed, previous)
                )

                namespace[name] = value

            return frozen.blind_v412_phase_recovery(
                crop,
                top_k=5,
                phase_step=16,
                canonical_size=256,
            )

        finally:
            for namespace, name, existed, previous in reversed(
                original_values
            ):
                if existed:
                    namespace[name] = previous
                else:
                    namespace.pop(name, None)
