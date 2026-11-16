# Reproducibility and scope (K-FRAG V4.15)

## What is present

This ZIP contains V4.15 neural model weights, embedding, regional decoding and refinement, source/authentication symbol scoring, Reed--Solomon processing, blind placement ranking, public verification APIs, frozen settings, dataset identifier lists, and synthetic integration scripts. The included `results/*.json` files are **transcriptions of historical reported aggregates**, not newly executed experiments. No original experimental key or trial-level crop/image outputs are distributed.

## Crucial difference between the two verifier pipelines

`kfrag_runtime.verify.verify_crop_v415()` calls `blind_crop_search_spatial_fast()` and then `recover_v415_at_placement()` on ranked candidates, with early return on an accepted candidate. It **does not call** the `blind_v412_phase_recovery()` named in the frozen research configuration. The historical notebook-derived entry point is preserved in `kfrag_runtime/frozen_runtime_functions.py` but not wired into the public verifier. Its phase-top-5 / 16-pixel offset expansion is not thereby reproduced. The public API is **not shown to be experimentally equivalent** to the historical benchmark; do not claim otherwise.

The uploaded code also does not provide the private evaluation key, original watermarked benchmark images, historical per-image crop coordinate manifest, original source-ID registry, or a standalone script that regenerates all 800 historical results. The `splits/` files identify COCO populations but do not establish independence from all earlier training stages. Matching filename-list hashes, if available, does not authenticate original image bytes.

## Run what is actually available

From the repository root with the published requirements installed and trusted checkpoints:

```bash
python tests/test_release_contract.py
python tests/test_public_runtime.py
python tests/test_public_roundtrip.py
# Slow, GPU preferred:
python tests/test_public_crop_recovery.py
python tests/test_public_retained_areas.py
```

`test_release_contract.py` checks checkpoint hashes and core protocol/configuration invariants; it does **not** execute the model. The other tests run deterministic synthetic carriers using a **public, non-secret** test key. Passing them does not validate the manuscript's DIV2K benchmark, negative controls, or matched baseline experiment. The retained-area script uses **centered square crops** with actual areas different from the manuscript's eight nominal crop settings.

## Historical DIV2K benchmark

100 validation images (0801--0900), 8 nominal retained-area settings = 800 blind crop trials. Reported correct source-verification counts: 83/100 at 25%, 99/100 at 30%, and 100/100 at 40%, 50%, 60%, 75%, 80%, and 95%. No incorrect-source acceptance was reported for those 800 trials. Exact crop origin need not be recovered for source verification to succeed. See `results/historical_div2k_blind_crop_aggregates.json` for the reported summary and its provenance caveats.

**Historical DIV2K Per-Image Evaluation Records**

`results/kfrag_v415_div2k_historical_per_image_results.csv` contains the original K-FRAG V4.15 blind-cropping evaluation outcomes for 100 DIV2K validation images (0801–0900) across eight nominal retained-area settings, totaling 800 trials.

The records were extracted from the archived experimental console logs, with exact duplicate entries removed and aggregate authentication and localization counts verified against the original results.

Each entry records the image identifier, nominal retained area, authentication outcome, exact crop-origin recovery status, and logged verification runtime. Runtimes retain the precision of the historical console output.

These are archived experimental results, not newly executed inference results.

## Matched-geometry baseline comparison (historical)

The manuscript reports WAM and InvisMark evaluated on the same 100 DIV2K validation images and identical physical crop coordinates and dimensions at eight nominal retained areas, with 800 crop configurations per baseline checked against K-FRAG's regenerated manifest. **The actual per-trial baseline CSVs and crop manifest are not in this uploaded release**, so the geometry matches cannot be independently inspected from these files. The reported percentages are transcribed in `results/reported_matched_crop_baselines.json`.

Different success criteria remain: K-FRAG counts keyed 48-bit source-token verification (including a soft-consistency fallback); WAM counts exact native 32-bit message recovery; InvisMark counts exact native 100-bit recovery. Therefore equal crop geometry does not imply a common authentication task.

## Separate negative controls and distortion tests

A separate 25%-area negative-control experiment reports 89/100 genuine fragments accepted and 100/100 unwatermarked fragments rejected. The incorrect-source control was a **post-verification comparison** against a cyclically shifted reference source, not a false identity presented to the verifier: 0/100 recovered tokens matched that incorrect reference. Do not combine the 89/100 figure with the main benchmark's 83/100, or treat the incorrect-reference test as proof of false-claim rejection.

Distortion results are for **known spatial alignment** and full images, not compound blind crop-and-distortion recovery. Reported JPEG Q95 is 98% and JPEG Q90 is 37%; the reported three Gaussian-noise conditions each reach 100%. The COCO 25% 96/100 result in the old README used an unresolved-only +/-16-pixel **rescue-assisted** procedure and is not a main V4.15 no-rescue result (`rescue_wrapper_in_final_method=false`).

## What the green visualization means

Highlighted regions depict distinct positions supplying coded **reconstruction support**. They do not establish independently authenticated regions, image-content integrity, or tamper-localization truth. The 48-bit recovered token is not the original 64-bit upstream identity; an external registry is needed to resolve the latter. The soft-consistency fallback checks expected keyed authentication symbols in prediction candidate sets without independently reconstructing the complete tag; formal security guarantees for this path have not been demonstrated. Transfer, replay and adaptive forgery resistance are unestablished.

## Key and checkpoint safety

Create a new random 32-byte authentication key for experiments. `bytes(range(32))` in tests is public. Never publish the original evaluation key. Model checkpoints use PyTorch deserialization; only load verified trusted artifacts and see `SECURITY.md`.

## Anonymous review

The **public GitHub repository is identifying** and must not be linked as the submitted contribution in anonymous CVPR 2027 materials. The separate code bundle is intended to be self-contained, with the author citation metadata removed, but it still requires final human inspection of checkpoint metadata, image assets, and any additional attachments. CVPR 2027 expects anonymous supplementary files and generally disallows external links that expand submitted content. Consult the official CVPR 2027 author guidance before uploading.
