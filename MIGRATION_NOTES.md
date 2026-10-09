# Migration notes

This package was created from the current K-FRAG notebook while preserving the final architecture and protocol.

## Safe to copy into GitHub now

- `kfrag_cvpr/model.py`
- `kfrag_cvpr/protocol.py`
- `kfrag_cvpr/final_decoder.py`
- `kfrag_cvpr/checkpoint.py`
- `kfrag_cvpr/legacy_bridge.py`
- `requirements-demo.txt`
- `.env.example`
- `.gitignore` entries

## Do not publish yet

- your private authentication key
- encrypted/private backup archives
- any checkpoint whose release rights/size you have not decided

## Next integration task

Map the current repository's existing implementations of embedding, independent decoding, fusion, blind search, and strict recovery into a thin `kfrag.demo_runtime` adapter. This should be a wiring-only change; do not retrain, alter thresholds, or alter model layers.
