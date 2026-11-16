# Matched-crop baseline evaluation records

The three CSVs here record crop geometry and observed baseline outputs for the 100 DIV2K validation images `0801.png`–`0900.png` at eight **nominal** retained-area settings: 25%, 30%, 40%, 50%, 60%, 75%, 80%, and 95%.

- `kfrag_v415_crop_manifest.csv`: 800 image/area crop configurations, including top-left coordinate, width, height and actual retained area.
- `wam_kfrag_matched_results.csv`: 900 rows (800 matched crops and 100 intact controls); native 32-bit WAM exact-message recovery and bit-accuracy records.
- `invismark_matched_results.csv`: 900 rows (800 matched crops and 100 intact controls); native 100-bit InvisMark exact-message recovery, bit-accuracy records, and per-image payload bitstrings.

All 800 WAM and all 800 InvisMark crop records match the K-FRAG manifest on image identity, nominal area, crop origin, width, height, and realized retained area. The CSVs do **not** contain historical per-trial K-FRAG authentication decisions. The 800-crop baseline comparison excludes the intact controls.

| Nominal area | WAM exact-32 (N=100) | InvisMark exact-100 (N=100) | Mean actual crop area |
|---|---:|---:|---:|
| 25% | 94% | 0% | 25.00% |
| 30% | 97% | 0% | 30.00% |
| 40% | 98% | 22% | 40.01% |
| 50% | 96% | 72% | 49.98% |
| 60% | 99% | 85% | 60.00% |
| 75% | 94% | 94% | 73.03% |
| 80% | 97% | 94% | 77.68% |
| 95% | 91% | 96% | 84.46% |

**Interpretation:** Crop geometry is matched, but verification tasks are not equivalent. K-FRAG reports keyed source-token verification, whereas WAM and InvisMark report exact recovery of their native messages. Cropping and message criteria should not be conflated. The nominal area label and the realized pixel area are distinct, especially at 75%, 80%, and 95%.

**Reproduction limit:** The audit script verifies internal consistency of the uploaded result files and their crop matching. It does not execute WAM/InvisMark models, prove that the reported outputs came from their official checkpoints, or reproduce the historical K-FRAG blind-crop runs.

Run at repository root:

```powershell
python tests/audit_matched_crop_baselines.py
```

The source files were provided as recorded experimental outputs; only filenames were normalized for repository organization.
