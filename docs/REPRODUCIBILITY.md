**Reproducibility and scope (K-FRAG V4.15)**


**What is present**

This archive contains the frozen V4.15 neural checkpoints, embedding and decoding components, regional refinement, source/authentication symbol scoring, Reed-Solomon processing, blind spatial search, verification interfaces, experimental configuration, dataset identifiers, and regression tests.

The release also includes archived K-FRAG DIV2K per-image evaluation records, WAM and InvisMark matched-crop baseline CSV files, and a regenerated crop manifest. The historical results are archived experimental evidence, not newly executed benchmark results.

The original private authentication key, original watermarked evaluation images, and complete original experiment environment are not distributed.


**Verification interfaces and historical compatibility**

The release provides two distinct verification interfaces. The public `kfrag_runtime.verify.verify_crop_v415()` interface uses `blind_crop_search_spatial_fast()` to generate ranked placement candidates and invokes `recover_v415_at_placement()` for candidate recovery and authentication. The historical compatibility interface, `kfrag_runtime.historical_v415_adapter.recover_historical_v415()`, uses the preserved `blind_v412_phase_recovery()` phase-expanded search function.

The historical phase-search implementation retains a `recover_v410_at_placement` compatibility lookup name. The corrected adapter maps that lookup to V4.15 placement recovery through `recover_v415_at_placement()`. The historical single-crop regression passed source authentication, source-token reconstruction, exact placement recovery, and restoration of patched runtime globals for its tested synthetic crop.

These results establish successful operation of the tested historical adapter path. They do not independently establish equivalence between the separate public search interface and every evaluation condition in the original 800-trial benchmark. The original private evaluation key, original watermarked benchmark images, original source-ID registry, and a complete one-command reproduction of the historical 800 trials are not provided. The release includes an archived per-image evaluation CSV and regenerated matched-crop manifest, but these are not substitutes for the original private-key experimental setup.

The `splits/` files identify COCO subsets but do not by themselves establish independence from all earlier training stages. Filename-list hashes do not authenticate the original image bytes.

**Run what is actually available**

From the repository root with the published requirements installed and trusted checkpoints:

```bash

python tests/test_release_contract.py
python tests/audit_matched_crop_baselines.py

python -m tests.test_public_runtime
python -m tests.test_public_roundtrip
python -m tests.test_historical_v415_crop_recovery

# Additional tests; CUDA recommended:
python -m tests.test_public_crop_recovery
python -m tests.test_public_retained_areas

```

The release-contract test checks checkpoint SHA-256 hashes and configuration invariants without neural inference. The matched-crop baseline audit validates the recorded CSVs, including 800/800 matching crop configurations per baseline and 100 intact-image controls for each baseline. It does
not rerun WAM or InvisMark inference.

The public roundtrip and historical single-crop regression exercise neural inference using deterministic synthetic inputs and a public non-secret testing key. The historical regression passed authentication, source-token recovery, exact crop placement, and restoration of patched
runtime globals.

The retained-area test uses centered square crops rather than the historical benchmark's eight nominal crop configurations. These regression tests are not a fresh execution of the complete historical 800-trial DIV2K benchmark or its negative-control experiments.

**Historical DIV2K benchmark**

100 validation images (0801--0900), 8 nominal retained-area settings = 800 blind crop trials. Reported correct source-verification counts: 83/100 at 25%, 99/100 at 30%, and 100/100 at 40%, 50%, 60%, 75%, 80%, and 95%. No incorrect-source acceptance was reported for those 800 trials. Exact crop origin need not be recovered for source verification to succeed. See `results/historical_div2k_blind_crop_aggregates.json` for the reported summary and its provenance caveats.

**Historical DIV2K Per-Image Evaluation Records**

`results/kfrag_v415_div2k_historical_per_image_results.csv` contains the original K-FRAG V4.15 blind-cropping evaluation outcomes for 100 DIV2K validation images (0801–0900) across eight nominal retained-area settings, totaling 800 trials.

The records were extracted from the archived experimental console logs, with exact duplicate entries removed and aggregate authentication and localization counts verified against the original results.

Each entry records the image identifier, nominal retained area, authentication outcome, exact crop-origin recovery status, and logged verification runtime. Runtimes retain the precision of the historical console output.

These are archived experimental results, not newly executed inference results.


**Matched-geometry baseline comparison (historical)**

The manuscript reports WAM and InvisMark comparisons on the same 100 DIV2K validation images at eight nominal retained areas.

The release includes recorded per-trial WAM and InvisMark baseline CSVs and a regenerated K-FRAG crop manifest under `results/matched_crop_baselines/`.

The accompanying audit checks 800/800 matched crop configurations per baseline and 100 intact-image controls for each baseline. The CSV audit recomputes the recorded exact-message recovery counts:

| Area | WAM exact 32-bit | InvisMark exact 100-bit |
|---|---:|---:|
| 25% | 94/100 | 0/100 |
| 30% | 97/100 | 0/100 |
| 40% | 98/100 | 22/100 |
| 50% | 96/100 | 72/100 |
| 60% | 99/100 | 85/100 |
| 75% | 94/100 | 94/100 |
| 80% | 97/100 | 94/100 |
| 95% | 91/100 | 96/100 |

This audit validates the consistency of recorded results and crop geometry. It does not rerun the baseline neural models or independently authenticate the original experiment.

The success criteria differ: K-FRAG counts keyed 48-bit source-token verification, including its documented soft-consistency fallback; WAM counts exact native 32-bit message recovery; and InvisMark counts exact native 100-bit message recovery. Matched crop geometry therefore does not establish that these methods perform an identical authentication task.


**Separate negative controls and distortion tests**

A separate 25%-area negative-control experiment reports 89/100 genuine fragments accepted and 100/100 unwatermarked fragments rejected. The incorrect-source control was a **post-verification comparison** against a cyclically shifted reference source, not a false identity presented to the verifier: 0/100 recovered tokens matched that incorrect reference. Do not combine the 89/100 figure with the main benchmark's 83/100, or treat the incorrect-reference test as proof of false-claim rejection.

Distortion results are for **known spatial alignment** and full images, not compound blind crop-and-distortion recovery. Reported JPEG Q95 is 98% and JPEG Q90 is 37%; the reported three Gaussian-noise conditions each reach 100%. The COCO 25% 96/100 result in the old README used an unresolved-only +/-16-pixel **rescue-assisted** procedure and is not a main V4.15 no-rescue result (`rescue_wrapper_in_final_method=false`).

**What the green visualization means**

Highlighted regions depict distinct positions supplying coded **reconstruction support**. They do not establish independently authenticated regions, image-content integrity, or tamper-localization truth. The 48-bit recovered token is not the original 64-bit upstream identity; an external registry is needed to resolve the latter. The soft-consistency fallback checks expected keyed authentication symbols in prediction candidate sets without independently reconstructing the complete tag; formal security guarantees for this path have not been demonstrated. Transfer, replay and adaptive forgery resistance are unestablished.

**Key and checkpoint safety**

Create a new random 32-byte authentication key for experiments. `bytes(range(32))` in tests is public. Never publish the original evaluation key. Model checkpoints use PyTorch deserialization; only load verified trusted artifacts and see `SECURITY.md`.

**Anonymous review**

The **public GitHub repository is identifying** and must not be linked as the submitted contribution in anonymous CVPR 2027 materials. The separate code bundle is intended to be self-contained, with the author citation metadata removed, but it still requires final human inspection of checkpoint metadata, image assets, and any additional attachments. CVPR 2027 expects anonymous supplementary files and generally disallows external links that expand submitted content. Consult the official CVPR 2027 author guidance before uploading.
