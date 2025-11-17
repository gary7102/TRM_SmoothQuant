# smoothquant_trm.py
#
# SmoothQuant + W8A8 fake quantization for TRM (TinyRecursiveModels).
# This module is intentionally self-contained and non-intrusive:
# - it does not modify original model code
# - it only uses hooks and state_dict operations
#
# Usage:
#   from smoothquant_trm import apply_smoothquant
#   sq_info = apply_smoothquant(model, alpha=0.7, calib_loader=calib_loader, device=device)
#
# After this call, model will run with:
#   - weights: offline fake-quantized to int8 (W8)
#   - activations: forward_pre_hook doing SmoothQuant scaling + int8 fake quant (A8)

from typing import Dict, Any, Iterable, List, Tuple, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 0. Helper: CastedLinear type
# ---------------------------------------------------------------------------

try:
    # In TinyRecursiveModels, CastedLinear is defined in models.layers
    from models.layers import CastedLinear  # type: ignore
except Exception:
    # Fallback for testing outside the repo
    CastedLinear = nn.Linear  # type: ignore


# ---------------------------------------------------------------------------
# 1. Generic helpers for batch handling and forward
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move a nested batch (dict/list/tuple/tensor) to target device."""
    if batch is None:
        return None
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    if isinstance(batch, (list, tuple)):
        converted = []
        for x in batch:
            converted.append(move_batch_to_device(x, device))
        return type(batch)(converted)
    return batch


def forward_step(model: nn.Module, batch: Any) -> Any:
    """Call model with batch (dict -> **batch, tuple/list -> *batch)."""
    if batch is None:
        return model()
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, (list, tuple)):
        return model(*batch)
    return model(batch)


def run_calibration(
    model: nn.Module,
    calib_loader: Iterable,
    device: torch.device,
    max_batches: int = 256,
) -> None:
    """Run a few forward passes over calib_loader (hooks do the real work)."""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if i >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            _ = forward_step(model, batch)


# ---------------------------------------------------------------------------
# 2. Find all target linear modules (CastedLinear)
# ---------------------------------------------------------------------------

def find_target_linears(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Find all linear-like modules to be quantized.
    Currently: all instances of CastedLinear.
    """
    targets: Dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        if isinstance(m, CastedLinear):
            targets[name] = m
    return targets


# ---------------------------------------------------------------------------
# 3. First pass: collect activation max BEFORE folding scales
# ---------------------------------------------------------------------------

def collect_activation_max(
    model: nn.Module,
    target_linears: Dict[str, nn.Module],
    calib_loader: Iterable,
    device: torch.device,
    max_batches: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    For each target linear, collect per-channel max(|X_j|) of its input.
    X is flattened to (N, C) and per-channel max is taken over N.
    """
    act_max: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach()
            # shape: (..., C) -> (N, C)
            x = x.view(-1, x.size(-1))
            cur = x.abs().amax(dim=0).to("cpu")  # (C,)
            if name not in act_max:
                act_max[name] = cur
            else:
                act_max[name] = torch.maximum(act_max[name], cur)
        return hook

    for name, m in target_linears.items():
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    run_calibration(model, calib_loader, device, max_batches=max_batches)

    for h in handles:
        h.remove()

    # Ensure all tensors on CPU float32
    for k in list(act_max.keys()):
        act_max[k] = act_max[k].to("cpu", dtype=torch.float32)
    return act_max


# ---------------------------------------------------------------------------
# 4. Compute SmoothQuant scales from act_max + weight_max
# ---------------------------------------------------------------------------

def compute_smoothquant_scales(
    model: nn.Module,
    target_linears: Dict[str, nn.Module],
    act_max: Dict[str, torch.Tensor],
    alpha: float = 0.7,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-input-channel scales s_j for each linear:
        s_j = (max_x_j^alpha) / (max_w_j^(1-alpha))
    Assumes weight shape (out_features, in_features).
    """
    sq_scales: Dict[str, torch.Tensor] = {}
    state = model.state_dict()

    for name, m in target_linears.items():
        if name not in act_max:
            continue
        w_name = f"{name}.weight"
        if w_name not in state:
            continue
        W = state[w_name].detach().to("cpu", dtype=torch.float32)  # (out, in)
        max_w = W.abs().amax(dim=0) + eps  # (in,)
        max_x = act_max[name].float() + eps  # (in,)
        s = (max_x ** alpha) / (max_w ** (1.0 - alpha))  # (in,)
        sq_scales[name] = s
    return sq_scales


# ---------------------------------------------------------------------------
# 5. Fold scales into weights and store inv_scale on modules
# ---------------------------------------------------------------------------

def fold_scales_into_weights(
    model: nn.Module,
    target_linears: Dict[str, nn.Module],
    sq_scales: Dict[str, torch.Tensor],
) -> None:
    """
    For each linear:
      W := W * s (per input-channel)
    and store inv_scale = 1/s as a buffer on the module for later use.
    """
    with torch.no_grad():
        for name, m in target_linears.items():
            if name not in sq_scales:
                continue
            s = sq_scales[name]  # (in,)
            if not torch.is_tensor(m.weight):
                continue
            device = m.weight.device
            s_dev = s.to(device, dtype=m.weight.dtype)  # (in,)
            # W: (out, in), broadcast multiply over in-dim
            m.weight.mul_(s_dev)
            inv_s = (1.0 / s).to("cpu", dtype=torch.float32)
            # store inv_scale as buffer; used by activation pre-hook
            if hasattr(m, "sq_inv_scale"):
                # overwrite if existed
                delattr(m, "sq_inv_scale")
            m.register_buffer("sq_inv_scale", inv_s)


# ---------------------------------------------------------------------------
# 6. Second pass: collect activation scales AFTER folding (for A8)
# ---------------------------------------------------------------------------

def collect_activation_scale_after_fold(
    model: nn.Module,
    target_linears: Dict[str, nn.Module],
    calib_loader: Iterable,
    device: torch.device,
    max_batches: int = 256,
) -> Dict[str, float]:
    """
    After folding S into weights, collect activation max(|X * S^{-1}|).
    For simplicity, we use symmetric per-tensor activation scale:
        delta_x = max_abs / 127
    Returns dict[name -> delta_x].
    """
    act_scale: Dict[str, float] = {}
    handles: List[Any] = []

    def make_hook(name: str, module: nn.Module):
        def hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach()
            # move sq_inv_scale to x.device
            if not hasattr(module, "sq_inv_scale"):
                return
            inv_s = module.sq_inv_scale.to(x.device, dtype=x.dtype)  # (C,)
            # shape: (..., C) -> (N, C)
            x = x.view(-1, x.size(-1))
            x_scaled = x * inv_s  # (N, C)
            max_abs = float(x_scaled.abs().max().item())
            if name not in act_scale:
                act_scale[name] = max_abs
            else:
                act_scale[name] = max(act_scale[name], max_abs)
        return hook

    for name, m in target_linears.items():
        if not hasattr(m, "sq_inv_scale"):
            continue
        h = m.register_forward_hook(make_hook(name, m))
        handles.append(h)

    run_calibration(model, calib_loader, device, max_batches=max_batches)

    for h in handles:
        h.remove()

    # convert max_abs -> delta_x
    for name, max_abs in list(act_scale.items()):
        if max_abs < 1e-8:
            max_abs = 1e-8
        act_scale[name] = max_abs / 127.0
    return act_scale


# ---------------------------------------------------------------------------
# 7. Offline fake quantization of weights to int8 (symmetric per-out-channel)
# ---------------------------------------------------------------------------

def fake_quantize_weights_int8(
    model: nn.Module,
    target_linears: Dict[str, nn.Module],
) -> None:
    """
    For each linear:
      - compute per-output-channel symmetric scale
      - quantize/dequantize weight once (W8 fake quant)
      - store weight scales as buffer for potential hardware export
    """
    with torch.no_grad():
        for name, m in target_linears.items():
            if not torch.is_tensor(m.weight):
                continue
            W = m.weight.data  # (out, in)
            # per-output-channel max
            max_abs = W.abs().amax(dim=1)  # (out,)
            max_abs = torch.clamp(max_abs, min=1e-8)
            scale = (max_abs / 127.0).view(-1, 1)  # (out, 1)
            q = torch.round(W / scale).clamp(-127, 127)
            W_q = q * scale
            m.weight.data = W_q
            # store weight scale if needed later
            w_scale = (max_abs / 127.0).to("cpu", dtype=torch.float32)
            if hasattr(m, "sq_w_scale"):
                delattr(m, "sq_w_scale")
            m.register_buffer("sq_w_scale", w_scale)


# ---------------------------------------------------------------------------
# 8. Attach activation fake quant hooks (SmoothQuant + A8)
# ---------------------------------------------------------------------------

def attach_activation_fake_quant(
    model: nn.Module,
    target_linears: Dict[str, nn.Module],
    act_scale: Dict[str, float],
    device: torch.device,
) -> List[Any]:
    """
    Attach forward_pre_hooks to each linear:
      X -> X * S^{-1} -> int8 quant/dequant (per-tensor symmetric)
    Returns list of handles; caller is responsible for removing them later.
    """
    handles: List[Any] = []

    def make_pre_hook(name: str, module: nn.Module):
        def pre_hook(mod: nn.Module, inputs: Tuple[Any, ...]):
            if not inputs:
                return None
            x = inputs[0]
            if not torch.is_tensor(x):
                return None

            x_dev = x
            if not hasattr(module, "sq_inv_scale"):
                return None

            inv_s = module.sq_inv_scale.to(x_dev.device, dtype=x_dev.dtype)  # (C,)
            delta_x_val = act_scale.get(name, 1.0)
            delta_x = torch.tensor(delta_x_val, device=x_dev.device, dtype=x_dev.dtype)

            # shape: (..., C)
            x_flat = x_dev.view(-1, x_dev.size(-1))
            x_scaled = x_flat * inv_s  # (N, C)
            # symmetric per-tensor int8 quant/dequant
            q = torch.round(x_scaled / delta_x).clamp(-127, 127)
            x_q = (q * delta_x).view_as(x_dev)

            new_inputs = (x_q,) + inputs[1:]
            return new_inputs
        return pre_hook

    for name, m in target_linears.items():
        if name not in act_scale:
            continue
        if not hasattr(m, "sq_inv_scale"):
            continue
        h = m.register_forward_pre_hook(make_pre_hook(name, m))
        handles.append(h)

    return handles


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------

def apply_smoothquant(
    model: nn.Module,
    alpha: float,
    calib_loader: Iterable,
    device: Optional[torch.device] = None,
    max_calib_batches: int = 256,
) -> Dict[str, Any]:
    """
    High-level pipeline:
      1) find target linears (CastedLinear)
      2) first pass: collect act_max (no scaling)
      3) compute per-channel SmoothQuant scales s_j
      4) fold s_j into weights and store inv_s on modules
      5) second pass: collect activation scales after folding (delta_x)
      6) offline fake-quant weights to int8
      7) attach activation fake-quant hooks (SmoothQuant + A8)

    Returns:
      {
        "sq_scales": dict[name -> Tensor(in,)],
        "act_scale": dict[name -> float],
        "fq_handles": list[hook_handle]
      }
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.to(device)

    # 1) find all target linear layers
    target_linears = find_target_linears(model)

    # 2) first pass: collect per-channel activation max
    act_max = collect_activation_max(
        model=model,
        target_linears=target_linears,
        calib_loader=calib_loader,
        device=device,
        max_batches=max_calib_batches,
    )

    # 3) compute SmoothQuant scales
    sq_scales = compute_smoothquant_scales(
        model=model,
        target_linears=target_linears,
        act_max=act_max,
        alpha=alpha,
    )

    # 4) fold scales into weights and store inv_scale
    fold_scales_into_weights(
        model=model,
        target_linears=target_linears,
        sq_scales=sq_scales,
    )

    # 5) second pass: collect activation scales after folding
    act_scale = collect_activation_scale_after_fold(
        model=model,
        target_linears=target_linears,
        calib_loader=calib_loader,
        device=device,
        max_batches=max_calib_batches,
    )

    # 6) offline fake quantize weights to int8
    fake_quantize_weights_int8(
        model=model,
        target_linears=target_linears,
    )

    # 7) attach activation fake quant hooks
    fq_handles = attach_activation_fake_quant(
        model=model,
        target_linears=target_linears,
        act_scale=act_scale,
        device=device,
    )

    return {
        "sq_scales": sq_scales,
        "act_scale": act_scale,
        "fq_handles": fq_handles,
    }

