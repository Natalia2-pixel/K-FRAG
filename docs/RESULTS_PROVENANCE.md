**Evidence and claim inventory**

This release includes archived K-FRAG V4.15 experimental records, matched-crop baseline evaluation files, frozen model checkpoints, implementation code, and regression tests.

The three `results/*.json` summary files contain author-reported historical aggregates transcribed from the manuscript and earlier research records. They are not newly generated experimental results.

The release additionally includes the historical DIV2K per-image evaluation CSV and per-trial WAM/InvisMark baseline CSV files. These records are distinct from the aggregate JSON summaries.

The public tests verify selected implementation properties and synthetic recovery cases. They do not constitute a new execution of the complete 800-trial DIV2K benchmark.

**Evidence inventory**

| Claim / evidence | Included? | Qualification |
|---|---|---|
| Frozen V4.15 configuration | Yes | Preserved configuration and historical search settings |
| Neural model checkpoints | Yes | Three checkpoint files passed SHA-256 integrity checks |
| Public deterministic synthetic tests | Yes | Uses public testing keys, not the original private evaluation key |
| Historical V4.15 single-crop regression | Yes | Passed source authentication, token recovery, exact placement, and runtime-global restoration |
| 800 DIV2K per-image evaluation records | Yes | Archived CSV records, not newly executed neural inference |
| Original DIV2K physical crop images | No | Original watermarked benchmark image crops are not distributed |
| Regenerated crop manifest | Yes | Included for matched-geometry consistency auditing; not the original private-key experiment |
| WAM per-trial baseline CSV | Yes | 800/800 matched crop configurations; recorded outcomes audited |
| InvisMark per-trial baseline CSV | Yes | 800/800 matched crop configurations; recorded outcomes audited |
| Baseline intact-image controls | Yes | 100 recorded controls for each baseline |
| Original private evaluation key | No | Intentionally excluded |
| Original source-ID registry | No | Not publicly distributed |
| Negative-control raw trial outcomes | No | Separate reported aggregate counts only |
| COCO train/dev/final filename lists | Yes | Does not independently establish all-stage training isolation |
| Historical phase-expanded recovery adapter | Yes | Routes preserved phase search to V4.15 placement recovery |
| Complete rerunnable historical 800-trial benchmark | No | Original private key, benchmark image artifacts, and complete historical execution setup are unavailable |

**Verification interfaces**

The release contains two distinct verification interfaces.

`kfrag_runtime.verify.verify_crop_v415()` uses the public candidate-ranking and V4.15 placement-recovery path. `kfrag_runtime.historical_v415_adapter.recover_historical_v415()`
uses the preserved phase-expanded spatial search and dispatches placement recovery to
`recover_v415_at_placement()` through the historical compatibility lookup.

The historical single-crop regression passed source authentication, correct source-token recovery, exact crop placement, and restoration of runtime globals on its tested synthetic crop.

This is evidence of successful execution of the corrected historical adapter. It is not an independent reproduction of all 800 manuscript evaluation trials.

Equivalence between the separate public `verify_crop_v415()` search interface and the complete
frozen phase-expanded evaluation has not been independently established.

**Matched-crop baseline evidence**

The included WAM and InvisMark baseline CSV records were checked against the regenerated crop manifest. The consistency audit passed 800/800 crop configurations for each baseline and identified 100 intact-image controls per baseline. Recorded exact-message recovery counts match the manuscript:

| Nominal retained area | WAM exact 32-bit | InvisMark exact 100-bit |
|---|---:|---:|
| 25% | 94/100 | 0/100 |
| 30% | 97/100 | 0/100 |
| 40% | 98/100 | 22/100 |
| 50% | 96/100 | 72/100 |
| 60% | 99/100 | 85/100 |
| 75% | 94/100 | 94/100 |
| 80% | 97/100 | 94/100 |
| 95% | 91/100 | 96/100 |

This audit checks recorded CSV consistency and geometry. It does not rerun WAM or InvisMark neural inference. The K-FRAG, WAM, and InvisMark columns use different success criteria and should not be interpreted as a common cryptographic authentication benchmark.

**Scope and limitations**

- Archived results are distinguished from newly executed tests.
- The original private evaluation key is excluded.
- The 800-trial DIV2K benchmark has not been freshly rerun.
- A correct source identity does not require exact crop-origin recovery.
- Reconstruction-support regions are not independently authenticated.
- The V4.15 soft-consistency fallback does not independently
  reconstruct the full authentication tag.
- Negative-control findings and known-alignment distortion
  tests belong to separate evaluation protocols.
- Rescue-assisted COCO results are not part of the main
  no-rescue DIV2K benchmark.

**Release modifications**

For this submission preparation:

- The historical compatibility adapter was corrected to
  dispatch placement recovery to the V4.15 implementation.
- The static call-chain audit passed.
- The release-contract and checkpoint integrity checks passed.
- The public end-to-end roundtrip test passed.
- The historical V4.15 single-crop regression passed.
- The matched-crop baseline CSV consistency audit passed.
- Documentation was updated to reflect the files included
  and the distinction between archived evidence and
  executable regression tests.

The submitted manuscript, frozen model checkpoints, experimental configuration, and reported results were not modified during these release corrections. No complete 800-trial neural benchmark was rerun as part of this documentation and packaging update.
