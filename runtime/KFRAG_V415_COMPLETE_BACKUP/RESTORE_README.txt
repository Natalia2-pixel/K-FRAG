K-FRAG V4.15 COMPLETE BACKUP
=============================

Created:
2026-10-26T23:43:12.291041Z


PURPOSE
-------

This archive is intended to preserve the current K-FRAG V4.15 research
runtime independently of the live Kaggle notebook.

It includes:

1. Historical V4 source backup tree, if mounted.
2. V4.15 symbol-head weights.
3. State dictionaries for all detected global torch.nn.Module objects.
4. Active K-FRAG runtime source snapshot.
5. V4.13 / V4.14 / V4.15 recovery code.
6. Frozen search and recovery configuration.
7. Dataset split manifests.
8. Environment and package versions.
9. SHA256 checksums.


SECURITY
--------

V3H_TRAIN_KEY IS NOT INCLUDED.

Only SHA256(V3H_TRAIN_KEY) is stored.

The real key must be restored separately from the secure location used by
the project.


DATA
----

The COCO images themselves are NOT copied into this ZIP because the image
dataset is very large.

The split manifests ARE included so the same image population can be
reconstructed as long as the dataset is available.


CURRENT V4.15 RECOVERY STACK
----------------------------

256 x 256 canonical image
4 x 4 spatial grid
16 regions
64 x 64 per region

V4:
RS(16,6)
6-byte source token
6-byte authentication tag

Blind synchronization:
COARSE_STRIDE = 2
TOP_COARSE    = 60
REFINE_RADIUS = 10

Phase refinement:
TOP_K      = 5
PHASE_STEP = 16

Recovery order:

1. Original Top-6 recovery
2. Coupled 6-of-8 subset recovery
3. Decoupled source/auth 6-of-8 recovery
4. Soft-auth V4.15 fallback

Soft-auth settings:

source_pool     = 8
auth_pool       = 8
auth_top_k      = 4
min_auth_support= 6


LATEST DEVELOPMENT/DEBUG RESULT
-------------------------------

25% retained, 10 actively tuned development images:

Correct authenticated = 10 / 10
Wrong authenticated   = 0 / 10
Unresolved            = 0 / 10
Ambiguous             = 0 / 10

Method breakdown:

original_top6       = 6
subset_6_of_8       = 1
decoupled_6_of_8    = 1
soft_auth           = 2


IMPORTANT SCIENTIFIC NOTE
-------------------------

These 10 images were used while developing V4.13-V4.15.

Therefore 10/10 IS NOT an unbiased final result.

Do not report this as final paper performance.


NEXT VALIDATION
---------------

1. Restore this exact V4.15 configuration.
2. Do not retrain.
3. Do not modify thresholds.
4. Use fresh, untouched development images.
5. First run 100 images at 25% retained.
6. Measure:
   - correct authenticated recovery
   - wrong authenticated recovery
   - unresolved
   - ambiguous
   - method breakdown
   - runtime
7. If satisfactory, run retained-area sweep:
   95, 80, 75, 60, 50, 40, 30, 25 percent.
8. Do not touch locked-final images until method selection is frozen.


RESTORE AUDIT
-------------

Missing required runtime entries at backup time:

[]


RESTORE PRINCIPLE
-----------------

The archive should be restored together with:

- the COCO dataset
- the separately stored private HMAC key

The private key should be verified against:

config/PRIVATE_KEY_FINGERPRINT_ONLY.txt
