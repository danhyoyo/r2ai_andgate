from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def sentence_transformer_device(
    *,
    purpose: str,
    env_var: str = "",
    min_free_gb_env: str = "",
    min_free_gb: float | None = None,
) -> str | None:
    """Return an explicit device for sentence-transformers when CUDA is unsafe.

    Recent PyTorch wheels used by Kaggle may omit kernels for Tesla P100
    (compute capability sm_60). In that case torch.cuda.is_available() is true,
    but the first model forward crashes with cudaErrorNoKernelImageForDevice.

    The caller can also force a device with env vars such as R2AI_RERANK_DEVICE=cpu.
    """
    requested = os.environ.get(env_var) if env_var else ""
    requested = requested or os.environ.get("R2AI_ST_DEVICE", "")
    if requested and requested.lower() != "auto":
        return requested

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

    threshold = _float_env(min_free_gb_env, min_free_gb or 0.0) if min_free_gb_env else (min_free_gb or 0.0)
    if threshold > 0:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
        except Exception as exc:
            print(f"CUDA memory probe failed for {purpose}; using CPU: {exc}")
            return "cpu"
        if free_gb < threshold:
            print(
                f"CUDA free memory is {free_gb:.2f} GiB / {total_gb:.2f} GiB before {purpose}; "
                f"threshold is {threshold:.2f} GiB. Using CPU for {purpose}."
            )
            return "cpu"
    return None
