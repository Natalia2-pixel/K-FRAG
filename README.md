# K-FRAG

**K-FRAG V4.15 — Fragment-Based Authenticated Watermarking with Blind Crop Recovery**

K-FRAG is a spatially distributed image-watermarking and authentication system designed for robust provenance recovery from complete images and cropped image fragments.

The public V4.15 release includes the neural runtime, trained model checkpoints, authenticated V4 watermark issuance and embedding, fragment decoding, structure-aware fusion, regional refinement, Reed–Solomon recovery, cryptographic authentication, blind spatial synchronization, and public reproducibility tests.

---

## Overview

K-FRAG distributes authenticated watermark information across a 4 × 4 spatial partition of a 256 × 256 RGB image.

Each of the 16 spatial fragments carries a 20-bit V4 payload:

```text
4 bits   spatial/index information
8 bits   source-symbol information
8 bits   authentication-symbol information
------------------------------------------
20 bits  per spatial fragment
```

The system combines:

- spatially distributed watermark embedding,
- independent fragment decoding,
- structure-aware fragment fusion,
- regional refinement,
- weak-symbol recovery,
- authentication-symbol recovery,
- Reed–Solomon error/erasure recovery,
- HMAC-based authentication,
- blind crop synchronization,
- authenticated token reconstruction.

The public release does **not** include the private authentication key used in the original research experiments.

---

## Repository Structure

```text
K-FRAG/
├── checkpoints/
│   ├── KFRAG_V415_FINAL_MODELS.pt
│   ├── fusion_model_state_dict.pt
│   └── v2_model_audit_state_dict.pt
│
├── config/
│   └── frozen public configuration and release metadata
│
├── docs/
│   └── supporting documentation
│
├── environment/
│   └── environment and dependency information
│
├── kfrag_runtime/
│   ├── models/
│   ├── runtime_loader.py
│   ├── runtime_ops.py
│   ├── protocol_v4.py
│   ├── embedding.py
│   ├── recovery.py
│   └── verify.py
│
├── manifests/
│   └── release and integrity metadata
│
├── runtime/
│   └── preserved runtime/recovery material
│
├── splits/
│   └── reproducibility split information
│
├── tests/
│   ├── test_public_runtime.py
│   ├── test_public_roundtrip.py
│   ├── test_public_crop_recovery.py
│   └── test_public_retained_areas.py
│
├── .gitignore
├── LICENSE
├── CITATION.cff
├── README.md
└── SECURITY.md
```

---

## Model Parameter Counts

The released K-FRAG V4.15 runtime contains the following neural components:

| Component | Parameters |
|---|---:|
| V2 watermark encoder (`StageDTagCapacityV1`) | 298,528 |
| V3 independent decoder | 213,268 |
| Structure-aware fragment fusion | 1,298,709 |
| Regional verification/refinement model | 437,885 |
| Weak-symbol residual head | 135,386 |
| Authentication-symbol residual head | 135,386 |
| **Total unique runtime parameters** | **2,519,162** |

The total is reported as the unique parameter count of the released runtime components.

---

## Requirements

The runtime requires Python and PyTorch together with the dependencies provided under the repository's environment configuration.

Important dependencies include:

```text
torch
torchvision
Pillow
reedsolo==1.7.0
```

Install the repository dependencies using the environment/requirements file supplied with the release.

---

## Loading the V4.15 Runtime

```python
from kfrag_runtime.runtime_loader import load_v415_runtime

runtime = load_v415_runtime(
    "checkpoints/KFRAG_V415_FINAL_MODELS.pt"
)

print(runtime["system"])
print(runtime["device"])
```

Expected system identifier:

```text
K-FRAG V4.15
```

The runtime automatically selects CUDA when available and otherwise runs on CPU.

---

## Authentication Keys

K-FRAG uses a 32-byte secret key for authenticated watermark issuance and verification.

The private key used for the original research experiments is **not distributed with this repository**.

Users must provide their own 32-byte authentication key:

```python
auth_key = YOUR_32_BYTE_SECRET_KEY

assert isinstance(auth_key, bytes)
assert len(auth_key) == 32
```

Do not:

- hard-code production authentication keys into source files,
- commit secret keys to Git,
- include secret keys in checkpoints,
- include secret-key paths in public tests,
- publish experimental private authentication material.

Keys should be supplied externally through an appropriate secret-management mechanism.

See `SECURITY.md` for additional guidance.

---

## Watermark Embedding

```python
from kfrag_runtime.runtime_loader import load_v415_runtime
from kfrag_runtime.embedding import embed_full_spatial_image_v4

runtime = load_v415_runtime(
    "checkpoints/KFRAG_V415_FINAL_MODELS.pt"
)

# Public example key only.
auth_key = bytes(range(32))

result = embed_full_spatial_image_v4(
    clean_image=image_tensor,
    source_id=b"example-source-id",
    runtime=runtime,
    auth_key=auth_key,
)

watermarked_image = result["watermarked"]
```

The current public embedding interface operates on 256 × 256 RGB images.

The released V4 embedding configuration uses the restored trained V2 watermark encoder together with the authenticated 20-bit regional V4 payload.

---

## Verification

Complete images and cropped image fragments can be verified with:

```python
from kfrag_runtime.verify import verify_crop_v415
from kfrag_runtime.protocol_v4 import V4_AUTH_DOMAIN

result = verify_crop_v415(
    crop=image_or_crop,
    runtime=runtime,
    auth_key=auth_key,
    auth_domain=V4_AUTH_DOMAIN,
    top_k=20,
)

print("accepted:", result.get("accepted"))
print("status:", result.get("status"))
```

Successful verification performs authenticated recovery rather than relying solely on neural decoder confidence.

---

## Public Regression Tests

### Runtime and Embedding

```bash
python tests/test_public_runtime.py
```

Expected final line:

```text
PUBLIC REGRESSION TEST: PASS
```

This checks:

- runtime restoration,
- trained V2 encoder loading,
- V4 fragment issuance,
- 20-bit fragment construction,
- watermark embedding.

---

### End-to-End Authenticated Round Trip

```bash
python tests/test_public_roundtrip.py
```

Expected final line:

```text
PUBLIC END-TO-END ROUNDTRIP TEST: PASS
```

The test validates:

```text
runtime loading
→ V4 issuance
→ watermark embedding
→ neural decoding
→ recovery
→ cryptographic authentication
→ recovered-token match
```

A deterministic non-secret test key is used.

---

### Blind Crop Recovery

```bash
python tests/test_public_crop_recovery.py
```

Expected final line:

```text
PUBLIC BLIND CROP RECOVERY TEST: PASS
```

This test exercises blind spatial synchronization and authenticated recovery from a cropped watermarked image.

Authentication success and spatial-placement accuracy are reported separately.

---

## Retained-Area Evaluation

The slower integration evaluation can be run with:

```bash
python tests/test_public_retained_areas.py
```

The public deterministic evaluation currently tests:

| Crop Size | Linear Retention | Retained Area |
|---|---:|---:|
| 256 × 256 | 100.00% | 100.00% |
| 240 × 240 | 93.75% | 87.89% |
| 224 × 224 | 87.50% | 76.56% |
| 208 × 208 | 81.25% | 66.02% |
| 192 × 192 | 75.00% | 56.25% |
| 176 × 176 | 68.75% | 47.27% |
| 160 × 160 | 62.50% | 39.06% |

For the deterministic public regression carrier, authenticated token recovery succeeded for all of the above tested crop sizes.

These values are provided as **software reproducibility checks** and should not be interpreted as replacements for large-scale experimental results on the research evaluation dataset.

---

## Fresh-Clone Reproducibility

The public repository has been tested from a fresh GitHub clone using:

```bash
python tests/test_public_runtime.py
python tests/test_public_roundtrip.py
python tests/test_public_crop_recovery.py
python tests/test_public_retained_areas.py
```

These tests confirm that the released V4.15 public runtime does not depend on hidden notebook state or an already initialized experimental runtime.

---

## Checkpoints

The public runtime requires:

```text
checkpoints/
├── KFRAG_V415_FINAL_MODELS.pt
├── fusion_model_state_dict.pt
└── v2_model_audit_state_dict.pt
```

### `KFRAG_V415_FINAL_MODELS.pt`

Contains the final V4.15 neural recovery components, including the V3 independent decoder, regional verification/refinement model, weak-symbol head, and authentication-symbol head.

### `fusion_model_state_dict.pt`

Contains the trained structure-aware fragment-fusion model.

### `v2_model_audit_state_dict.pt`

Contains the restored trained V2 watermark encoder required for V4 watermark embedding.

---

## CPU and GPU Execution

K-FRAG supports both CPU and CUDA execution.

The lightweight regression tests can run on CPU.

Blind spatial synchronization is substantially more computationally expensive because multiple candidate placements are scored and refined. CUDA is therefore recommended for larger crop evaluations and benchmarking.

---

## Security

The original experimental authentication key is not included in this repository.

The public tests use non-secret deterministic keys strictly for software validation.

See:

```text
SECURITY.md
```

for key-handling and deployment guidance.

---

## Reproducibility Scope

This release is intended to reproduce the K-FRAG V4.15 software pipeline, including:

- V4 authenticated payload issuance,
- spatial fragment embedding,
- fragment decoding,
- structure-aware fusion,
- regional refinement,
- weak-symbol recovery,
- authentication-symbol recovery,
- Reed–Solomon reconstruction,
- HMAC authentication,
- blind crop synchronization,
- authenticated crop recovery.

Large-scale quantitative results reported in the associated research work should be reproduced using the corresponding dataset, evaluation population, crop protocol, retained-area settings, and experimental configuration.

---

## Citation

If you use K-FRAG in academic work, please cite this repository and the associated paper when available.

Citation metadata is provided in:

```text
CITATION.cff
```

---

## License

This project is released under the Apache License 2.0.

See:

```text
LICENSE
```

for the full license terms.
