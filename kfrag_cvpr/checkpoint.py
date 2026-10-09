"""Checkpoint integrity helpers for the final K-FRAG refiner."""

from pathlib import Path
import hashlib
import torch

from .model import KFRAGRegionalRefinerV2

FINAL_MODEL_STATE_SHA256 = "2cefcf9abfc087c3c2e8960a42f156e28bce6592f082c74f872fcd8474d60deb"
FINAL_CHECKPOINT_FILE_SHA256 = "3b4c0bb39eb09bcff2ad45d17f8993bb640cb1adc0fb66cdaafd4e7c783373c5"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state_dict(model):
    h = hashlib.sha256()
    for name in sorted(model.state_dict().keys()):
        tensor = model.state_dict()[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def load_final_refiner(checkpoint_path, device, verify_file_sha=True):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if verify_file_sha:
        actual_file_sha = sha256_file(checkpoint_path)
        if actual_file_sha != FINAL_CHECKPOINT_FILE_SHA256:
            raise RuntimeError(
                "Final checkpoint file SHA mismatch. "
                f"Expected {FINAL_CHECKPOINT_FILE_SHA256}, got {actual_file_sha}"
            )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = KFRAGRegionalRefinerV2(input_bits=20, image_features=64, hidden=128)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    actual_model_sha = sha256_state_dict(model)
    if actual_model_sha != FINAL_MODEL_STATE_SHA256:
        raise RuntimeError(
            "Final model-state SHA mismatch. "
            f"Expected {FINAL_MODEL_STATE_SHA256}, got {actual_model_sha}"
        )
    return model
