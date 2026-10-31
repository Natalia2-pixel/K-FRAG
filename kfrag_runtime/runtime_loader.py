from pathlib import Path

import torch

from .models import (
    NaturalChannelV2,
    RegionalChannelV1,
    StageD12BitTransitionV1,
    StageDTagCapacityV1,
    V3IndependentDecoder,
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
    Reconstruct the V2 hierarchy required to instantiate
    V3IndependentDecoder.

    The final V4.15 checkpoint stores the complete V3 decoder state,
    so this V2 graph is used as an architectural scaffold only.
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


def load_v415_runtime(
    checkpoint_path,
    device="auto",
):
    """
    Restore the frozen K-FRAG V4.15 neural runtime.

    Returns a dictionary containing:

        v3_independent_decoder
        verification_model
        weak_symbol_head
        auth_symbol_head
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
    # V3 independent decoder
    # --------------------------------------------------------

    v2_scaffold = _build_v2_scaffold()

    v3_independent_decoder = V3IndependentDecoder(
        v2_scaffold
    )

    v3_independent_decoder.load_state_dict(
        checkpoint["v3_independent_decoder"],
        strict=True,
    )

    # V2 is no longer needed after V3 construction/loading.
    del v2_scaffold

    # --------------------------------------------------------
    # Final verification/refiner model
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

    # --------------------------------------------------------
    # Final weak-symbol head
    # --------------------------------------------------------

    weak_symbol_head = WeakSymbolResidualHead()

    weak_symbol_head.load_state_dict(
        checkpoint["weak_symbol_head"],
        strict=True,
    )

    # --------------------------------------------------------
    # Final authentication-symbol head
    # --------------------------------------------------------

    auth_symbol_head = WeakSymbolResidualHead()

    auth_symbol_head.load_state_dict(
        checkpoint["auth_symbol_head"],
        strict=True,
    )

    # --------------------------------------------------------
    # Freeze final runtime
    # --------------------------------------------------------

    v3_independent_decoder = _freeze(
        v3_independent_decoder,
        resolved_device,
    )

    verification_model = _freeze(
        verification_model,
        resolved_device,
    )

    weak_symbol_head = _freeze(
        weak_symbol_head,
        resolved_device,
    )

    auth_symbol_head = _freeze(
        auth_symbol_head,
        resolved_device,
    )

    return {
        "system": checkpoint["system"],
        "format_version": checkpoint["format_version"],
        "device": resolved_device,
        "v3_independent_decoder": v3_independent_decoder,
        "verification_model": verification_model,
        "weak_symbol_head": weak_symbol_head,
        "auth_symbol_head": auth_symbol_head,
    }