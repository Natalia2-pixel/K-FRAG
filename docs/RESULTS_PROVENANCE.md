# Evidence and claim inventory

The three `results/*.json` files are **author-reported historical aggregates** transcribed from the uploaded manuscript / README. They are not experimental outputs created by running this release. Each JSON states the origin and missing evidence.

| Claim / evidence | Included? | Qualification |
|---|---|---|
| Frozen V4.15 configuration | Yes | Records historical phase function; public verifier differs |
| Neural weights | Yes | Original checkpoint bytes preserved |
| Public deterministic synthetic tests | Yes | Tests not executed with manuscript's private key |
| 800 DIV2K per-trial records | No | Only reported aggregate counts |
| Original DIV2K crop coordinates / physical crops | No | Cannot recheck from ZIP |
| WAM/InvisMark per-trial baseline CSVs | No | Aggregate success percentages transcribed only |
| Original private key | No | Must remain private |
| Negative-control raw outcomes | No | Reported separate control counts only |
| COCO train/dev/final filename lists | Yes | Not an all-stages independence guarantee |
| Historical exact phase search executable benchmark | No | Preserved function is not a complete rerunnable suite |

## Changes made in this documentation update

- Added missing distinction between public crop-verifier and historical phase-expanded benchmark.
- Corrected baseline matching description and clarified different recovery tasks.
- Clarified incorrect-source reference test is post-verification.
- Separated rescue-assisted COCO results from the DIV2K no-rescue main protocol.
- Clarified green regional support is not independent regional authentication.
- Added reported aggregate JSONs, release-integrity check, and a CVPR anonymous-code package.

**Not changed:** original V4.15 neural model code, original training weights/checkpoints, and frozen configuration. No 800-image experiment was rerun in producing these documentation artifacts.
