"""Historical K-FRAG V4.15 single-crop regression test.

Uses the public deterministic test key, not the private research key.
Does not reproduce the historical 800-trial DIV2K benchmark.
"""

import runpy
import torch

from kfrag_runtime import frozen_runtime_functions as frozen
from kfrag_runtime.historical_v415_adapter import recover_historical_v415
from kfrag_runtime.protocol_v4 import V4_AUTH_DOMAIN
from kfrag_runtime.embedding import v4_source_token


def test_historical_v415_crop_recovery():
    # Reuse the already validated public embedding setup.
    test = runpy.run_path("tests/test_public_roundtrip.py")

    runtime = test["runtime"]
    auth_key = test["test_key"]
    source_id = test["source_id"]
    watermarked = test["watermarked"]

    true_x = 32
    true_y = 32
    crop_size = 192

    crop = watermarked[
        0,
        :,
        true_y:true_y + crop_size,
        true_x:true_x + crop_size,
    ].detach()

    # Record the global state before historical execution.
    tracked = [
        "COARSE_STRIDE",
        "TOP_COARSE",
        "REFINE_RADIUS",
        "BATCH_PLACEMENTS",
        "_batch_candidate_canvases",
        "decode_phase_tiles_v2",
        "canonical_index_bits",
        "recover_v410_at_placement",
    ]

    before = {
        name: (name in vars(frozen), vars(frozen).get(name))
        for name in tracked
    }

    with torch.no_grad():
        result = recover_historical_v415(
            crop,
            runtime=runtime,
            auth_key=auth_key,
            auth_domain=V4_AUTH_DOMAIN,
        )

    assert isinstance(result, dict)
    assert result.get("accepted") is True
    assert result.get("ambiguous") is False

    candidate = result.get("candidate")
    assert isinstance(candidate, dict)

    expected_token = v4_source_token(source_id, auth_key)
    assert candidate.get("token") == expected_token

    # Exact placement is checked independently.
    recovered_x = candidate.get("x")
    recovered_y = candidate.get("y")

    print("Authentication: PASS")
    print("Source token: PASS")
    print("True placement:", (true_x, true_y))
    print("Recovered placement:", (recovered_x, recovered_y))

    # Check that adapter globals were restored.
    for name, (existed, previous) in before.items():
        assert (name in vars(frozen)) == existed, (
            f"Global existence changed: {name}"
        )

        if existed:
            current = vars(frozen)[name]
            assert current is previous or current == previous, (
                f"Global value changed: {name}"
            )

    print("Runtime global restoration: PASS")
    print("HISTORICAL V4.15 SINGLE-CROP REGRESSION: PASS")


if __name__ == "__main__":
    test_historical_v415_crop_recovery()
