# Security

K-FRAG V4.15 uses a private HMAC key.

The following must never be committed:

- PRIVATE_v3_hmac_key.pt
- raw V3H_TRAIN_KEY bytes
- secret environment files
- private backups containing secret material

The public repository contains only public model/runtime artifacts.
