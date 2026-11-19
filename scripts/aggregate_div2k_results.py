
"""Aggregate K-FRAG V4.15 public DIV2K JSON evaluation records."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "image",
    "retained_area_pct",
    "authentication_outcome",
    "accepted",
    "ambiguous",
    "token_matches",
    "true_x",
    "true_y",
    "recovered_x",
    "recovered_y",
    "exact_xy",
    "tested_placements",
    "recovery_runtime_seconds",
    "device",
    "evaluation_key_type",
    "source_id_algorithm",
    "timestamp_utc",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="results/public_replication",
    )

    parser.add_argument(
        "--output",
        default=(
            "results/public_replication/"
            "div2k_v415_public_evaluations.csv"
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    rows = []
    seen = set()

    for path in sorted(input_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))

        required = {
            "image",
            "retained_area_pct",
            "authentication_outcome",
            "accepted",
            "ambiguous",
            "token_matches",
            "true_xy",
            "recovered_xy",
            "exact_xy",
            "tested_placements",
            "recovery_runtime_seconds",
            "device",
            "evaluation_key_type",
            "source_id_algorithm",
            "timestamp_utc",
        }

        missing = required - record.keys()

        if missing:
            raise ValueError(
                f"{path}: missing fields {sorted(missing)}"
            )

        if record["evaluation_key_type"] != (
            "public_deterministic_test_key"
        ):
            raise ValueError(
                f"{path}: unexpected evaluation key type"
            )

        key = (
            record["image"],
            record["retained_area_pct"],
        )

        if key in seen:
            raise ValueError(
                f"Duplicate image/area evaluation: {key}"
            )

        seen.add(key)

        true_xy = record["true_xy"]
        recovered_xy = record["recovered_xy"]

        if not isinstance(true_xy, list) or len(true_xy) != 2:
            raise ValueError(f"{path}: invalid true_xy")

        if recovered_xy is not None and (
            not isinstance(recovered_xy, list)
            or len(recovered_xy) != 2
        ):
            raise ValueError(f"{path}: invalid recovered_xy")

        rows.append({
            "image": record["image"],
            "retained_area_pct": record["retained_area_pct"],
            "authentication_outcome": record["authentication_outcome"],
            "accepted": record["accepted"],
            "ambiguous": record["ambiguous"],
            "token_matches": record["token_matches"],
            "true_x": true_xy[0],
            "true_y": true_xy[1],
            "recovered_x": (
                recovered_xy[0]
                if recovered_xy is not None
                else ""
            ),
            "recovered_y": (
                recovered_xy[1]
                if recovered_xy is not None
                else ""
            ),
            "exact_xy": record["exact_xy"],
            "tested_placements": record["tested_placements"],
            "recovery_runtime_seconds": (
                record["recovery_runtime_seconds"]
            ),
            "device": record["device"],
            "evaluation_key_type": record["evaluation_key_type"],
            "source_id_algorithm": (
                record["source_id_algorithm"]
            ),
            "timestamp_utc": record["timestamp_utc"],
        })

    if not rows:
        raise RuntimeError("No evaluation JSON files found.")

    rows.sort(
        key=lambda row: (
            row["retained_area_pct"],
            row["image"],
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(
        row["authentication_outcome"] == "correct_authenticated"
        for row in rows
    )

    exact = sum(
        row["exact_xy"] is True
        for row in rows
    )

    print("=" * 72)
    print("K-FRAG V4.15 PUBLIC DIV2K AGGREGATION")
    print("=" * 72)
    print("JSON records:", len(rows))
    print("Correct authenticated:", correct)
    print("Exact XY:", exact)
    print("Output:", output_path)
    print("AGGREGATION: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
