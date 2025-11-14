"""Shared utilities for tensor-aware components."""

try:
    import torch
except ImportError as exc:  # pragma: no cover - clarity during optional installs
    torch = None
    _torch_import_error = exc
else:
    _torch_import_error = None


def get_torch_device():
    """Return the preferred torch device, raising if torch is unavailable."""

    if torch is None:
        raise ImportError(
            "PyTorch is required for tensor-based execution but is not installed."
        ) from _torch_import_error
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = ["get_torch_device"]
