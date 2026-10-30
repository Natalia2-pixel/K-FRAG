K-FRAG V4 BACKUP
=================

CURRENT ARCHITECTURE
--------------------
Canonical image: 256x256
Grid: 4x4 = 16 fragments
Fragment payload: 20 bits
    4 index bits
    8 source bits
    8 auth bits

V4:
    RS(16,6) source
    RS(16,6) auth
    48-bit source token
    48-bit HMAC-derived authentication tag

IMPORTANT PIPELINE FIX
----------------------
Old embed_full_spatial_image() used:

    v3_issue(...)

This caused V3 embedding / V4 recovery mismatch.

Correct V4 embedding is now:

    embed_full_spatial_image_v4(...)

It uses:

    v4_issue(...)

Packet consistency was confirmed:

    embed target == v4_issue: True

V4.10 RESULT
------------
Fresh true-V4 25% crop:

    Oracle authentication : PASS
    Blind authentication  : PASS
    Ambiguous             : False
    Predicted XY          : exact true XY
    Spatial rank          : 1

V4.11 PARTIAL BENCHMARK
-----------------------
40% retained, 200 dev images:

    correct authenticated : 164 / 200 = 82.0%
    wrong authenticated   : 0
    unresolved            : 36 / 200 = 18.0%
    ambiguous             : 0

This is BELOW the desired >90% recovery target.

NEXT TASK
---------
Do NOT retrain immediately.

Analyze the 36 unresolved 40% cases and separate:

1. true XY missing from blind Top-20
2. true/near-true XY present but fixed top-six visibility subset fails
3. alternative six-region subset could authenticate
4. symbol-head failure despite correct placement

Likely next optimization:
adaptive region-subset recovery + improved spatial candidate coverage.

SECURITY
--------
Do not weaken HMAC.
Do not accept approximate auth.
Exact HMAC verification remains mandatory.

IMPORTANT
---------
Do not use the old V3 embed_full_spatial_image() for V4 benchmarks.

Use:

    embed_full_spatial_image_v4(...)
