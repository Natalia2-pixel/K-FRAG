# K-FRAG CVPR public-release candidate

This folder is a **safe modularization of the current K-FRAG final model**, not a redesign.

## What is exact

- `kfrag_cvpr/model.py`: final `MultiScaleRegionalEncoder` and `KFRAGRegionalRefinerV2` layer definitions.
- `kfrag_cvpr/protocol.py`: V3 RS(16,8) source/authentication construction and HMAC-SHA256 protocol.
- `kfrag_cvpr/final_decoder.py`: final 100K fusion rule: `base_fused + fragment_logits20 - raw_mean`.
- `kfrag_cvpr/checkpoint.py`: final model/checkpoint integrity hashes and strict state loading.

## What is deliberately NOT reimplemented

The notebook restores several earlier K-FRAG components from the existing repository/backups rather than defining them from scratch in the final notebook. To avoid changing the CVPR system, this bundle does not invent replacements for:

- watermark embedding (`embed_full_spatial_image`)
- base independent decoder
- existing fusion model
- blind spatial search internals
- strict authenticated recovery internals
- legacy `v3_fragments_to_bits20` helper

Use `kfrag_cvpr/legacy_bridge.py` to connect those existing validated components.

## Final frozen values

- Canonical image: 256x256 RGB
- Grid: 4x4, 16 regions of 64x64
- Regional payload: 20 bits = 4 index + 8 source + 8 authentication
- Strict authenticated-fragment requirement: 8
- Coarse stride: 4
- Top coarse candidates: 20
- Refinement radius: 6 pixels
- Top strict candidates: 20
- Final refiner parameters: 437,885
- Final model-state SHA256: `2cefcf9abfc087c3c2e8960a42f156e28bce6592f082c74f872fcd8474d60deb`
- Final checkpoint-file SHA256: `3b4c0bb39eb09bcff2ad45d17f8993bb640cb1adc0fb66cdaafd4e7c783373c5`

## Demo

`demo/app.py` is a Gradio frontend. It intentionally requires an adapter to your existing validated runtime. This avoids silently replacing or simplifying any part of the model before CVPR submission.

Do not commit the real authentication key. Store it as a deployment secret.

## Important release note

Before calling the repository fully reproducible, add the exact legacy base-model/embedding/search/recovery source files from your existing GitHub repository and publish a compatible final checkpoint. This candidate already preserves the final regional-refiner architecture and cryptographic protocol without changing them.
