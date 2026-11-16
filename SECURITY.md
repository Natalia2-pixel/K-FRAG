# Security and key handling

The original research HMAC key is deliberately absent. Generate a new, cryptographically random 32-byte key for private experiments. Do not use the deterministic `bytes(range(32))` key from public tests outside software demonstration.

Never commit key bytes, local secrets, secret-management paths, private notebook outputs, or original author-enrollment/identity data. The original evaluation key cannot be reconstructed from this release. The tool's authentication is source-token-level; the fallback path has no established formal security guarantee, nor have transfer, replay and adaptive forgery attacks been exhaustively evaluated.

**Checkpoint safety:** Only load authenticated/trusted model checkpoints. PyTorch deserialization, especially `torch.load(weights_only=False)`, can execute untrusted code. Verify checkpoint SHA-256 digests with `python tests/test_release_contract.py` and obtain checkpoints only from sources you trust.

**Anonymity:** The public repository includes identifying metadata; use a separately checked code-only anonymous bundle for review, not a URL to the public repository.
