## Public Runtime Regression Test

K-FRAG includes a public regression test that checks whether the restored V4.15 runtime can be loaded and used for watermark issuance and embedding without requiring access to the private research authentication key.

From the repository root, run:

```bash
python tests/test_public_runtime.py
```

A successful installation should end with:

```text
PUBLIC REGRESSION TEST: PASS
```

The test verifies that:

- the K-FRAG V4.15 runtime loads successfully,
- the trained V2 watermark encoder checkpoint is restored,
- the V4 issuance protocol produces 16 authenticated fragments,
- the fragment representation has shape `(16, 20)`,
- the watermark embedding path returns a `256 × 256` RGB image.

The regression test uses a deterministic public test key only for software validation. It is not the private authentication key used in the original research experiments.

## Authentication Keys

K-FRAG uses a 32-byte authentication key for authenticated watermark issuance and verification.

Authentication keys are intentionally **not stored in this repository**.

Applications should provide their key externally at runtime:

```python
from kfrag_runtime.runtime_loader import load_v415_runtime
from kfrag_runtime.embedding import embed_full_spatial_image_v4

runtime = load_v415_runtime(
    "checkpoints/KFRAG_V415_FINAL_MODELS.pt"
)

auth_key = YOUR_32_BYTE_SECRET_KEY

result = embed_full_spatial_image_v4(
    clean_image=image_tensor,
    source_id=b"example-source-id",
    runtime=runtime,
    auth_key=auth_key,
)

watermarked_image = result["watermarked"]
```

The key must be exactly 32 bytes:

```python
assert isinstance(auth_key, bytes)
assert len(auth_key) == 32
```

For security, do not:

- hard-code production authentication keys in source files,
- commit authentication keys to Git,
- include private keys in checkpoints,
- include private-key paths or values in public tests or examples.

A deployment may instead load the authentication key from an environment variable, secret manager, protected configuration file, or another external secret-storage mechanism.

## Core Runtime Files

The restored V4.15 public runtime uses the following checkpoint files:

```text
checkpoints/
├── KFRAG_V415_FINAL_MODELS.pt
├── fusion_model_state_dict.pt
└── v2_model_audit_state_dict.pt
```

The principal runtime modules are:

```text
kfrag_runtime/
├── runtime_loader.py
├── runtime_ops.py
├── protocol_v4.py
├── embedding.py
├── recovery.py
└── verify.py
```

`runtime_loader.py` restores the trained neural components required by the V4.15 pipeline.

`embedding.py` implements authenticated V4 issuance and watermark embedding.

`recovery.py` implements fragment recovery and authenticated token reconstruction.

`verify.py` provides the higher-level crop verification interface.

## Minimal Runtime Check

A minimal runtime-loading check can be performed with:

```python
from kfrag_runtime.runtime_loader import load_v415_runtime

runtime = load_v415_runtime(
    "checkpoints/KFRAG_V415_FINAL_MODELS.pt"
)

print(runtime["system"])
print(runtime["device"])
print(type(runtime["v2_model_audit"]).__name__)
```

Expected output includes:

```text
K-FRAG V4.15
StageDTagCapacityV1
```

The exact device may be `cpu` or `cuda` depending on the available hardware.
