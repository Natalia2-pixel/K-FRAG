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
