
**K-FRAG V4.15 — Anonymous Supplementary Code Information**

This supplementary-code archive contains the K-FRAG V4.15 implementation, frozen pretrained checkpoints, experimental configuration, public verification interfaces, a historical
phase-expanded recovery adapter, regression tests, and archived experimental records. The archive includes historical DIV2K per-image evaluation records, WAM and InvisMark matched-crop baseline CSVs, and a regenerated crop manifest.

The archived records document the 800 trials conducted for the manuscript. They are preserved experimental results, not outputs generated again during supplementary-code packaging. See `README.md`, `docs/REPRODUCIBILITY.md`, and `docs/RESULTS_PROVENANCE.md` for details.

**Installation**

Use Python 3.12 and install dependencies from `requirements.txt`, with a platform-compatible PyTorch and torchvision build.

```bash
python -m pip install -r requirements.txt
```

CPU execution is supported for smaller tests, but blind spatial search may be computationally expensive. CUDA is recommended for crop-recovery experiments.

**Run the supplied checks**

From the repository root:

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

The release-contract test checks checkpoint hashes and configuration without neural inference. The matched-crop baseline audit checks recorded CSV consistency, including 800 matching crop configurations per baseline and 100 intact controls per baseline. It does not rerun WAM or InvisMark neural inference.

The public roundtrip test exercises embedding and verification with a synthetic input. The historical single-crop regression exercises the phase-expanded recovery adapter. In the tested case, authentication, source-token recovery, exact placement, and runtime-global restoration all passed. The original K-FRAG V4.15 evaluation comprised 800 executed blind-crop trials on 100 DIV2K validation images across eight nominal retained-area conditions.

The corresponding historical per-image evaluation records are included in this release.

The regression tests supplied with the code validate selected implementation paths independently. They are not intended to rerun the original 800-trial evaluation.


**Verification interfaces**

The release provides two distinct interfaces:

- `kfrag_runtime.verify.verify_crop_v415()`:
  the public ranked-candidate verification interface.

- `kfrag_runtime.historical_v415_adapter.recover_historical_v415()`:
  the preserved phase-expanded search interface, with
  placement recovery dispatched to
  `recover_v415_at_placement()`.

The actual recovery callback is the V4.15 implementation.

The separate public verifier has not been independently established as experimentally equivalent to the complete frozen phase-search evaluation.

**Historical evidence and limitations**

The archive includes:

- Archived K-FRAG DIV2K per-image records for 800 trials.
- Recorded WAM and InvisMark baseline trial CSVs.
- A regenerated matched-crop manifest.
- Historical aggregate result summaries.

The original private evaluation key, original watermarked benchmark crops, original source-ID registry, and complete private-key evaluation setup are not supplied. The supplied regression tests are not substitutes for a full reproduction of all manuscript experiments.

The baseline methods also use different success criteria: K-FRAG verifies a keyed source identity, while WAM and InvisMark results describe exact native-message recovery.

**Anonymous review and security**

This archive is intended for anonymous peer review.

Before submission, inspect all files, including checkpoint metadata and images, for author-identifying information, personal contact details, credentials, identifying repository links, and local development paths.

Do not include `.git` history or private authentication keys.

The public example key `bytes(range(32))` is only for testing and must not be treated as a deployment secret. Load PyTorch checkpoints only from trusted sources. See `SECURITY.md`.

Submit the code as an anonymous supplementary archive through the conference submission system, subject to the applicable submission requirements.
