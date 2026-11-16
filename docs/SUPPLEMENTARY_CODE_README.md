# K-FRAG V4.15 -- anonymous supplementary code information

The code bundle includes model definitions, pretrained weights, frozen V4.15 settings, synthetic runtime/integration checks, and **historical** reported-result summaries. It does not include a full reproduction of the paper's private-key experiments. Start with `docs/REPRODUCIBILITY.md` and `docs/RESULTS_PROVENANCE.md`.

## Execute

Use Python 3.12 and dependencies in `requirements.txt`, plus a compatible PyTorch/torchvision build (see `README.md`). CPU runs are supported for smaller tests; blind spatial search is expensive and a CUDA device is recommended.

```
python tests/test_release_contract.py
python tests/test_public_runtime.py
python tests/test_public_roundtrip.py
python tests/test_public_crop_recovery.py
python tests/test_public_retained_areas.py
```

Only the first test is dependency-light and checks file/configuration integrity; all others use trained neural runtime. No published benchmark reproduction is claimed.

## Known boundaries

The `verify_crop_v415` public path differs from the notebook `blind_v412_phase_recovery` frozen path. Original authentication key, historical 800-trial crop manifest, author baseline trial CSVs, original benchmark image payloads and full benchmark harness are not supplied. Do not interpret reported aggregates as rerun outputs.

## Privacy

The anonymized ZIP removes author citation metadata from the public code repository copy. Review the files prior to submission. It is **not** an external-link substitute; submit code as a file in supplementary materials following conference upload restrictions.
