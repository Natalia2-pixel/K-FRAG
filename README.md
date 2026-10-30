# K-FRAG V4.15

Frozen reproducible public release of K-FRAG V4.15.

## Protocol

- 256x256 canonical image
- 4x4 regional grid
- 16 regions
- 64x64 region size
- 20-bit regional fragment
  - 4-bit logical region index
  - 8-bit source RS symbol
  - 8-bit authentication RS symbol
- RS(16,6)
- 6-byte source token
- 6-byte authentication tag

## Frozen recovery

Final recovery:

    blind_v412_phase_recovery

Frozen search:

    BATCH_PLACEMENTS = 24
    COARSE_STRIDE = 2
    TOP_COARSE = 60
    REFINE_RADIUS = 10
    TOP_STRICT = 20

    PHASE_TOP_K = 5
    PHASE_STEP = 16

    V415_SOURCE_POOL = 8
    V415_AUTH_POOL = 8
    V415_AUTH_TOP_K = 4
    V415_MIN_AUTH_SUPPORT = 6

The unresolved rescue wrapper and post-authentication phase refinement
are not part of the final method.

## Security

The private HMAC key is not included in this repository.

The verifier must provision the key separately.

Never commit:

    PRIVATE_v3_hmac_key.pt
    V3H_TRAIN_KEY raw bytes
    secret environment files

## Reproducibility

The exact frozen runtime snapshot and model state dictionaries are included.

Dataset image data is not included. Split manifests contain portable image IDs.
