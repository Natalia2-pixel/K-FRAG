from pathlib import Path

import torch

from .models import (
    NaturalChannelV2,
    RegionalChannelV1,
    StageD12BitTransitionV1,
    StageDTagCapacityV1,
    V3IndependentDecoder,
    StructureAwareFragmentFusion,
    KFRAGRegionalRefinerV2,
    WeakSymbolResidualHead,
)


EXPECTED_SYSTEM = "K-FRAG V4.15"
EXPECTED_FORMAT_VERSION = 2

REQUIRED_CHECKPOINT_KEYS = {
    "format_version",
    "system",
    "verification_model",
    "weak_symbol_head",
    "auth_symbol_head",
    "v3_independent_decoder",
}


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return resolved


def _build_v2_scaffold():
    """
    Reconstruct the historical K-FRAG V2 model hierarchy.

    This architecture serves two separate purposes:

    1. A trained V2 instance is restored from
       v2_model_audit_state_dict.pt for watermark issuance/embedding.

    2. A separate V2 instance is used as the architectural parent required
       to construct V3IndependentDecoder before loading the final V3 state.
    """

    stage_b = NaturalChannelV2(
        image_size=64,
        width=16,
        activation="silu",
        mask_floor=0.25,
    )

    stage_c = RegionalChannelV1(
        stage_b=stage_b,
        image_size=64,
    )

    stage_d = StageD12BitTransitionV1(
        stage_c=stage_c,
    )

    v2_model = StageDTagCapacityV1(
        parent=stage_d,
    )

    return v2_model


def _freeze(model, device):
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _extract_state_dict(checkpoint, name):
    """
    Accept either a raw state_dict or a simple checkpoint wrapper.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"{name} checkpoint must contain a dictionary."
        )

    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

        if not isinstance(state, dict):
            raise TypeError(
                f"{name} checkpoint['state_dict'] must be a dictionary."
            )

        return state

    return checkpoint


def load_v415_runtime(
    checkpoint_path,
    device="auto",
    fusion_checkpoint_path=None,
    v2_checkpoint_path=None,
):
    """
    Restore the frozen K-FRAG V4.15 runtime.

    The returned runtime contains:

        v2_model_audit
            Trained historical V2 watermark encoder used for
            issuance/embedding.

        v3_independent_decoder
            Final V3 decoder restored from the V4.15 checkpoint.

        fusion_model
            Structure-aware fragment fusion model.

        verification_model
            Final regional/refinement model.

        weak_symbol_head
            Source-byte residual head.

        auth_symbol_head
            Authentication-byte residual head.

        device
        system
        format_version
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"K-FRAG checkpoint not found: {checkpoint_path}"
        )

    resolved_device = _resolve_device(device)

    # --------------------------------------------------------
    # Resolve auxiliary checkpoint paths
    # --------------------------------------------------------

    if fusion_checkpoint_path is None:
        fusion_checkpoint_path = (
            checkpoint_path.parent
            / "fusion_model_state_dict.pt"
        )
    else:
        fusion_checkpoint_path = Path(
            fusion_checkpoint_path
        )

    if not fusion_checkpoint_path.is_file():
        raise FileNotFoundError(
            "K-FRAG fusion checkpoint not found: "
            f"{fusion_checkpoint_path}"
        )

    if v2_checkpoint_path is None:
        v2_checkpoint_path = (
            checkpoint_path.parent
            / "v2_model_audit_state_dict.pt"
        )
    else:
        v2_checkpoint_path = Path(
            v2_checkpoint_path
        )

    if not v2_checkpoint_path.is_file():
        raise FileNotFoundError(
            "K-FRAG V2 encoder checkpoint not found: "
            f"{v2_checkpoint_path}"
        )

    # --------------------------------------------------------
    # Load main frozen V4.15 bundle
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "K-FRAG V4.15 checkpoint must contain a dictionary."
        )

    missing_top_level = (
        REQUIRED_CHECKPOINT_KEYS - set(checkpoint.keys())
    )

    if missing_top_level:
        raise RuntimeError(
            "K-FRAG V4.15 checkpoint is missing required entries: "
            + ", ".join(sorted(missing_top_level))
        )

    if checkpoint["system"] != EXPECTED_SYSTEM:
        raise RuntimeError(
            "Unexpected K-FRAG system identifier: "
            f"{checkpoint['system']!r}"
        )

    if checkpoint["format_version"] != EXPECTED_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported K-FRAG checkpoint format version: "
            f"{checkpoint['format_version']!r}"
        )

    # --------------------------------------------------------
    # Restore trained historical V2 watermark encoder
    # --------------------------------------------------------

    v2_state_raw = torch.load(
        v2_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    v2_state = _extract_state_dict(
        v2_state_raw,
        "K-FRAG V2 encoder",
    )

    v2_model_audit = _build_v2_scaffold()

    v2_model_audit.load_state_dict(
        v2_state,
        strict=True,
    )

    v2_model_audit = _freeze(
        v2_model_audit,
        resolved_device,
    )

    # --------------------------------------------------------
    # Construct V3 decoder using a separate V2 scaffold
    # --------------------------------------------------------

    v2_scaffold = _build_v2_scaffold()

    v3_independent_decoder = V3IndependentDecoder(
        v2_scaffold
    )

    v3_independent_decoder.load_state_dict(
        checkpoint["v3_independent_decoder"],
        strict=True,
    )

    v3_independent_decoder = _freeze(
        v3_independent_decoder,
        resolved_device,
    )

    # Do not use v2_scaffold for embedding.
    # It belongs only to the reconstructed V3 decoder hierarchy.

    # --------------------------------------------------------
    # Structure-aware fusion model
    # --------------------------------------------------------

    fusion_state_raw = torch.load(
        fusion_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    fusion_state = _extract_state_dict(
        fusion_state_raw,
        "K-FRAG fusion",
    )

    fusion_model = StructureAwareFragmentFusion()

    fusion_model.load_state_dict(
        fusion_state,
        strict=True,
    )

    fusion_model = _freeze(
        fusion_model,
        resolved_device,
    )

    # --------------------------------------------------------
    # Final verification / regional refinement model
    # --------------------------------------------------------

    verification_model = KFRAGRegionalRefinerV2(
        input_bits=20,
        image_features=64,
        hidden=128,
    )

    verification_model.load_state_dict(
        checkpoint["verification_model"],
        strict=True,
    )

    verification_model = _freeze(
        verification_model,
        resolved_device,
    )

    # --------------------------------------------------------
    # Weak-symbol source-byte head
    # --------------------------------------------------------

    weak_symbol_head = WeakSymbolResidualHead()

    weak_symbol_head.load_state_dict(
        checkpoint["weak_symbol_head"],
        strict=True,
    )

    weak_symbol_head = _freeze(
        weak_symbol_head,
        resolved_device,
    )

    # --------------------------------------------------------
    # Authentication-symbol byte head
    # --------------------------------------------------------

    auth_symbol_head = WeakSymbolResidualHead()

    auth_symbol_head.load_state_dict(
        checkpoint["auth_symbol_head"],
        strict=True,
    )

    auth_symbol_head = _freeze(
        auth_symbol_head,
        resolved_device,
    )

    # --------------------------------------------------------
    # Final runtime bundle
    # --------------------------------------------------------

    return {
        "system": checkpoint["system"],
        "format_version": checkpoint["format_version"],
        "device": resolved_device,
        "v2_model_audit": v2_model_audit,
        "v3_independent_decoder": v3_independent_decoder,
        "fusion_model": fusion_model,
        "verification_model": verification_model,
        "weak_symbol_head": weak_symbol_head,
        "auth_symbol_head": auth_symbol_head,
    }