K-FRAG V4.15 END-OF-SESSION STATUS
==================================

25% retained — 10-image development/debug subset:

Correct authenticated : 10/10 = 1.000
Wrong authenticated   : 0/10  = 0.000
Unresolved            : 0/10  = 0.000
Ambiguous             : 0/10  = 0.000

Recovery breakdown:
- original_top6        : 6
- subset_6_of_8        : 1
- decoupled_6_of_8     : 1
- soft_auth            : 2

Important:
These 10 images were actively used during decoder development.
They are NOT an unbiased final evaluation set.

Current frozen candidate stack:
1. Strong blind synchronization
   COARSE_STRIDE = 2
   TOP_COARSE = 60
   REFINE_RADIUS = 10

2. +/-16 phase refinement
   PHASE_TOP_K = 5
   PHASE_STEP = 16

3. Original Top-6 RS(16,6) recovery

4. Coupled 6-of-8 subset fallback

5. Decoupled source/auth 6-of-8 fallback

6. Soft-auth fallback
   source pool = 8
   auth pool = 8
   auth Top-k = 4
   minimum auth support = 6

Security:
- No true source identity is supplied to blind recovery.
- No true crop XY is supplied to blind recovery.
- Expected source token is used only for evaluation scoring.
- V3H_TRAIN_KEY is NOT stored in this backup.
- The private key must be restored separately from the existing secure runtime/source.

NEXT SESSION:
1. Restore validated V4 base runtime.
2. Restore weak/auth model heads.
3. Reinstall V4.13 -> V4.14 -> V4.15 recovery stack.
4. Verify configuration.
5. Run untouched 100-image development test at 25% retained.
6. Measure:
   - correct authenticated recovery
   - wrong authenticated recovery
   - unresolved
   - ambiguous
   - recovery-method breakdown
   - runtime
7. Do NOT use locked-final set yet.
8. Do NOT retrain unless fresh-dev evidence shows decoder limitation.
