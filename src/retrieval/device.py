from __future__ import annotations

import os


def sentence_transformer_device(*, purpose: str) -> str | None:
    """Return an explicit device for sentence-transformers when CUDA is unsafe.

    Recent PyTorch wheels used by Kaggle may omit kernels for Tesla P100
    (compute capability sm_60). In that case torch.cuda.is_available() is true,
    but the first model forward crashes with cudaErrorNoKernelImageForDevice.
    """
    if os.environ.get("R2AI_ALLOW_LEGACY_CUDA") == "1":
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception as exc:
        print(f"CUDA probe failed for {purpose}; using CPU: {exc}")
        return "cpu"
    if major < 7:
        print(
            f"CUDA device {name} has compute capability sm_{major}{minor}. "
            f"This PyTorch build may not support it; using CPU for {purpose}. "
            "Set R2AI_ALLOW_LEGACY_CUDA=1 only if you installed a compatible torch wheel."
        )
        return "cpu"
    return None
