
"""Single-image DIV2K evaluation for K-FRAG V4.15.

Uses the restored historical recovery adapter, recorded crop geometry,
and a PUBLIC demonstration key.

This is a fresh-key replication, not a bit-for-bit reproduction of
the original private-key DIV2K experiment.
"""

import argparse
import csv
import time
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import torch
from PIL import Image

from kfrag_runtime.runtime_loader import load_v415_runtime
from kfrag_runtime.embedding import (
    embed_full_spatial_image_v4,
    v4_source_token,
)
from kfrag_runtime.protocol_v4 import V4_AUTH_DOMAIN
from kfrag_runtime.historical_v415_adapter import (
    recover_historical_v415,
)


ROOT = Path(__file__).resolve().parents[1]

EVAL_VERSION = "KFRAG_EVAL_V1"


def evaluation_source_id(dataset_name, relative_identifier):
    """Reconstructed historical eight-byte source identifier."""
    payload = (
        f"{EVAL_VERSION}|{dataset_name}|{relative_identifier}"
    ).encode("utf-8")

    return hashlib.sha256(payload).digest()[:8]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", default="0801.png")
    parser.add_argument("--area", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving evaluation results as JSON.",
    )
    args = parser.parse_args()

    image_path = ROOT / "data" / "DIV2K_valid_HR" / args.image

    manifest_path = (
        ROOT
        / "results"
        / "matched_crop_baselines"
        / "kfrag_v415_crop_manifest.csv"
    )

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    with manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        matches = [
            row for row in csv.DictReader(file)
            if row["image"] == args.image
            and int(row["nominal_area_pct"]) == args.area
        ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one matching crop manifest record."
        )

    row = matches[0]

    x = int(row["crop_x"])
    y = int(row["crop_y"])
    w = int(row["crop_width"])
    h = int(row["crop_height"])

    print("=" * 72)
    print("K-FRAG V4.15 DIV2K SINGLE-IMAGE EVALUATION")
    print("=" * 72)
    print("Image:", args.image)
    print("Nominal retained area:", args.area)
    print("Crop geometry:", (x, y, w, h))

    runtime = load_v415_runtime(
        checkpoint_path=ROOT / "checkpoints" / "KFRAG_V415_FINAL_MODELS.pt",
        fusion_checkpoint_path=ROOT / "checkpoints" / "fusion_model_state_dict.pt",
        v2_checkpoint_path=ROOT / "checkpoints" / "v2_model_audit_state_dict.pt",
        device=args.device,
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_size = image.size
        image = image.resize(
            (256, 256),
            Image.Resampling.LANCZOS,
        )

        array = (
            np.asarray(image, dtype=np.float32).copy()
            / 255.0
        )

    clean = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(runtime["device"])
    )

    # Public test key. Never replace this with the private research key
    # in a file intended for publication.
    auth_key = bytes(range(32))

    # Provisional demonstration source identifier.
    # Not yet verified as the exact historical evaluation_source_id().
    source_id = evaluation_source_id(
        "KFRAG_V415_DIV2K_VALID_OOD",
        args.image,
    )

    expected_token = v4_source_token(source_id, auth_key)

    with torch.no_grad():
        embedded = embed_full_spatial_image_v4(
            clean_image=clean,
            source_id=source_id,
            runtime=runtime,
            auth_key=auth_key,
        )

    watermarked = embedded["watermarked"]

    crop = watermarked[
        0,
        :,
        y:y+h,
        x:x+w,
    ].detach()

    print("Original image size:", original_size)
    print("Canonical image:", tuple(watermarked.shape))
    print("Crop shape:", tuple(crop.shape))
    print("Device:", runtime["device"])

    started = time.perf_counter()

    with torch.no_grad():
        result = recover_historical_v415(
            crop,
            runtime=runtime,
            auth_key=auth_key,
            auth_domain=V4_AUTH_DOMAIN,
        )

    elapsed = time.perf_counter() - started

    candidate = result.get("candidate")
    accepted = bool(result.get("accepted", False))
    ambiguous = bool(result.get("ambiguous", False))

    token_matches = (
        isinstance(candidate, dict)
        and candidate.get("token") == expected_token
    )

    recovered_xy = (
        (candidate.get("x"), candidate.get("y"))
        if isinstance(candidate, dict)
        else None
    )

    exact_xy = recovered_xy == (x, y)

    if ambiguous:
        status = "ambiguous"
    elif accepted and token_matches:
        status = "correct_authenticated"
    elif accepted:
        status = "wrong_authenticated"
    else:
        status = "unresolved"

    print("\n" + "=" * 72)
    print("EVALUATION RESULT")
    print("=" * 72)
    print("Status:", status)
    print("Accepted:", accepted)
    print("Ambiguous:", ambiguous)
    print("Token matches:", token_matches)
    print("True placement:", (x, y))
    print("Recovered placement:", recovered_xy)
    print("Exact XY:", exact_xy)
    print("Tested placements:", result.get("tested_placements"))
    print("Recovery runtime (seconds):", round(elapsed, 3))
    print("=" * 72)

    # ------------------------------------------------------------
    # SAVE STRUCTURED EVALUATION RESULT
    # ------------------------------------------------------------

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluation_record = {
            "experiment": "KFRAG_V415_DIV2K",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "image": args.image,
            "retained_area_pct": args.area,
            "original_image_size": list(original_size),
            "canonical_size": [256, 256],
            "crop_x": x,
            "crop_y": y,
            "crop_width": w,
            "crop_height": h,
            "authentication_outcome": status,
            "accepted": accepted,
            "ambiguous": ambiguous,
            "token_matches": bool(token_matches),
            "true_xy": [x, y],
            "recovered_xy": (
                list(recovered_xy)
                if recovered_xy is not None
                else None
            ),
            "exact_xy": bool(exact_xy),
            "tested_placements": result.get(
                "tested_placements"
            ),
            "recovery_runtime_seconds": round(
                elapsed, 3
            ),
            "device": str(runtime["device"]),
            "evaluation_key_type": "public_deterministic_test_key",
            "historical_private_key_used": False,
            "source_id_algorithm": (
                "KFRAG_EVAL_V1_SHA256_TRUNC8_RECONSTRUCTED"
            ),
        }

        with output_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                evaluation_record,
                file,
                indent=2,
            )

        print("\nEvaluation result saved to:", output_path)



if __name__ == "__main__":
    main()
