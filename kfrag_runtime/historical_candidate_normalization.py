"""Original K-FRAG historical candidate normalization.

Extracted verbatim from kfrag-crop-adapt-v2 (52)(6).ipynb, cell 393.
No changes to the two function bodies.
"""

def _extract_xy_candidate(item):

    if item is None:
        return None

    # ----------------------------------------------------------------------------------------------
    # Dictionary result
    # ----------------------------------------------------------------------------------------------

    if isinstance(item, dict):

        x = None
        y = None

        for key in [
            "x",
            "candidate_x",
            "crop_x",
            "left",
        ]:
            if key in item:
                x = item[key]
                break

        for key in [
            "y",
            "candidate_y",
            "crop_y",
            "top",
        ]:
            if key in item:
                y = item[key]
                break

        if x is None or y is None:
            return None

        score = None

        for score_key in [
            "score",
            "spatial_score",
            "index_score",
            "confidence",
        ]:
            if score_key in item:
                score = item[score_key]
                break

        hard_alignment = item.get(
            "hard_alignment",
            None,
        )

        confidence = item.get(
            "confidence",
            None,
        )

        return {
            "x": int(x),
            "y": int(y),

            "search_score": (
                float(score)
                if score is not None
                else None
            ),

            "hard_alignment": (
                float(hard_alignment)
                if hard_alignment is not None
                else None
            ),

            "confidence": (
                float(confidence)
                if confidence is not None
                else None
            ),

            "raw": item,
        }

    # ----------------------------------------------------------------------------------------------
    # Tuple/list form
    # ----------------------------------------------------------------------------------------------

    if isinstance(
        item,
        (tuple, list),
    ):

        if len(item) < 2:
            return None

        if not isinstance(
            item[0],
            (int, float),
        ):
            return None

        if not isinstance(
            item[1],
            (int, float),
        ):
            return None

        score = None

        if (
            len(item) >= 3
            and isinstance(
                item[2],
                (int, float),
            )
        ):
            score = float(item[2])

        return {
            "x": int(item[0]),
            "y": int(item[1]),
            "search_score": score,
            "hard_alignment": None,
            "confidence": None,
            "raw": item,
        }

    return None

def normalize_blind_candidates(result):

    if result is None:
        return []

    candidate_container = result

    if isinstance(result, dict):

        candidate_keys = [
            "ranked",               # <<< CRITICAL FIX
            "candidates",
            "placements",
            "ranked_candidates",
            "results",
            "top_candidates",
            "hypotheses",
        ]

        found = False

        for key in candidate_keys:

            if key in result:

                candidate_container = result[key]
                found = True
                break

        if not found:

            # Maybe result itself is a candidate dictionary.
            one = _extract_xy_candidate(result)

            if one is not None:
                candidate_container = [result]
            else:
                return []

    if not isinstance(
        candidate_container,
        (list, tuple),
    ):
        candidate_container = [
            candidate_container
        ]

    normalized = []

    for item in candidate_container:

        candidate = _extract_xy_candidate(
            item
        )

        if candidate is not None:
            normalized.append(
                candidate
            )

    # ----------------------------------------------------------------------------------------------
    # Remove duplicate XY placements while preserving order.
    # ----------------------------------------------------------------------------------------------

    unique = []
    seen = set()

    for candidate in normalized:

        xy = (
            candidate["x"],
            candidate["y"],
        )

        if xy in seen:
            continue

        seen.add(xy)
        unique.append(candidate)

    return unique
