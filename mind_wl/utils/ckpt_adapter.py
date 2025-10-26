"""Helpers to adapt PyTorch I3D checkpoints for MindSpore."""

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from mindspore import Parameter, Tensor


# Pair-wise replacements from PyTorch naming to MindSpore naming.
_BN_NAME_REPLACEMENTS: Sequence[Tuple[str, str]] = (
    (".bn.running_mean", ".bn.bn2d.moving_mean"),
    (".bn.running_var", ".bn.bn2d.moving_variance"),
    (".bn.weight", ".bn.bn2d.gamma"),
    (".bn.bias", ".bn.bn2d.beta"),
)


def _map_param_name(name: str) -> Optional[str]:
    """Translate a PyTorch parameter name to its MindSpore counterpart."""

    if name.endswith("num_batches_tracked"):
        # MindSpore BatchNorm does not use this bookkeeping tensor.
        return None

    mapped = name
    for src, dst in _BN_NAME_REPLACEMENTS:
        if src in mapped:
            mapped = mapped.replace(src, dst)
            break
    return mapped


def _to_numpy(value) -> np.ndarray:
    """Convert tensors/arrays to a NumPy array without altering precision."""

    if hasattr(value, "data") and hasattr(value.data, "asnumpy"):
        return value.data.asnumpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    return np.asarray(value)


def convert_pytorch_state_dict(pt_state_dict: Dict[str, object]) -> Tuple[OrderedDict, List[str]]:
    """Convert a PyTorch state dict to a MindSpore-friendly parameter dict."""

    converted: OrderedDict = OrderedDict()
    skipped: List[str] = []

    for name, value in pt_state_dict.items():
        target_name = _map_param_name(name)
        if target_name is None:
            skipped.append(name)
            continue

        np_value = _to_numpy(value)
        converted[target_name] = Parameter(Tensor(np_value), name=target_name)

    return converted, skipped


def remap_mindspore_param_dict(param_dict: Dict[str, object]) -> Tuple[OrderedDict, List[str]]:
    """Remap an incompatible MindSpore checkpoint to the expected naming."""

    remapped: OrderedDict = OrderedDict()
    skipped: List[str] = []

    for name, value in param_dict.items():
        target_name = _map_param_name(name)
        if target_name is None:
            skipped.append(name)
            continue

        remapped[target_name] = Parameter(Tensor(_to_numpy(value)), name=target_name)

    return remapped, skipped


def needs_remap(param_names: Iterable[str]) -> bool:
    """Check whether the provided parameter names require BatchNorm remapping."""

    for name in param_names:
        if ".bn." in name and ".bn.bn2d." not in name:
            return True
    return False


def build_checkpoint_payload(param_dict: Dict[str, Parameter]) -> List[Dict[str, Tensor]]:
    """Create a MindSpore checkpoint payload from a Parameter dictionary."""

    payload: List[Dict[str, Tensor]] = []
    for name, param in param_dict.items():
        payload.append({"name": name, "data": Tensor(_to_numpy(param))})
    return payload
