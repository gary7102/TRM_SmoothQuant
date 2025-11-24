import torch
import torch.nn as nn
from dataclasses import is_dataclass, fields
from typing import Dict, Iterable, Optional

from models.layers import CastedLinear


class W8A8Linear(nn.Module):
    """
    模擬 W8A8 量化的 Linear 層。
    - 權重使用 per-input-channel fake quant (sym int8)
    - activation 使用 per-tensor fake quant (sym int8)
    - 支援 SmoothQuant 的 input scaling（每個 input channel 一個 scale）
    """

    def __init__(self, original_layer: CastedLinear, input_scale: Optional[torch.Tensor] = None):
        super().__init__()

        # 線性層維度
        self.in_features = original_layer.weight.shape[1]
        self.out_features = original_layer.weight.shape[0]

        # 拷貝權重 / bias
        self.weight = nn.Parameter(original_layer.weight.detach().clone())
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        # SmoothQuant input scaling s^{-1}，shape=[in_features]
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.view(-1).float())
        else:
            self.register_buffer("input_scale", torch.ones(self.in_features))

    @staticmethod
    def _fake_quant_tensor(
        x: torch.Tensor,
        num_bits: int = 8,
        per_channel: bool = False,
        channel_dim: int = 0,
    ):
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        x = x.float()

        if per_channel:
            # 沿 channel_dim 做 per-channel quant（例如 weight: [out, in], channel_dim=1 代表 per-input）
            abs_max = x.abs().amax(dim=channel_dim, keepdim=True).clamp(min=1e-5)
            scale = abs_max / qmax
            x_q = torch.round(x / scale).clamp(qmin, qmax)
            x_deq = x_q * scale
        else:
            abs_max = x.abs().amax().clamp(min=1e-5)
            scale = abs_max / qmax
            x_q = torch.round(x / scale).clamp(qmin, qmax)
            x_deq = x_q * scale

        return x_deq.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C_in]

        # 1) SmoothQuant input scaling
        if self.input_scale is not None:
            scale = self.input_scale.to(x.device).to(x.dtype)  # [C_in]
            # 建立可 broadcast 的 shape：例如 x=[B,L,C] → [1,1,C]；x=[B,C] → [1,C]
            shape = [1] * (x.dim() - 1) + [self.in_features]
            x = x * scale.view(*shape)

        # 2) Fake quant activation (per-tensor)
        x_q = self._fake_quant_tensor(x, num_bits=8, per_channel=False)

        # 3) Fake quant weight (per-input-channel; weight: [out, in], channel_dim=1)
        w_q = self._fake_quant_tensor(self.weight, num_bits=8, per_channel=True, channel_dim=1)

        # 4) 線性運算（用 FP 做模擬）
        out = torch.nn.functional.linear(x_q, w_q, self.bias)
        return out


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if is_dataclass(obj):
        kv = {}
        for f in fields(obj):
            kv[f.name] = _to_device(getattr(obj, f.name), device)
        return type(obj)(**kv)
    return obj


def _iter_take(iterator, max_batches: int):
    for i, item in enumerate(iterator):
        if i >= max_batches:
            break
        yield i, item


def collect_activation_and_weight_stats(
    model: nn.Module,
    calib_iterator,
    device: torch.device,
    max_batches: int = 32,
):
    """
    採樣 CastedLinear 的輸入 activation 絕對值最大值 (per-channel)。
    會回傳 dict[name] = torch.Tensor(shape=[in_features])
    """
    model.eval()
    act_max: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            if not inputs:
                return
            x = inputs[0].detach().to(torch.float32)
            # 不假設維度數；只假設最後一維是 channel
            x_flat = x.view(-1, x.shape[-1])  # [N, C_in]
            cur = x_flat.abs().amax(dim=0).cpu()
            if name in act_max:
                act_max[name] = torch.maximum(act_max[name], cur)
            else:
                act_max[name] = cur
        return hook

    # 掛 hook
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
            print(f"[quant_utils] Register hook on CastedLinear: {name}")

    with torch.no_grad():
        for batch_idx, sample in _iter_take(calib_iterator, max_batches):
            # PuzzleDataset 可能回傳 (set_name, batch, eff_bs)
            if isinstance(sample, tuple) and len(sample) == 3:
                _, batch, _ = sample
            else:
                batch = sample
            batch = {k: v.to(device) for k, v in batch.items()}

            carry = model.initial_carry(batch)
            carry = _to_device(carry, device)

            # 單步 forward 即可收集 activation
            _carry_out, _outputs = model(carry, batch)
            if batch_idx % 5 == 0:
                print(f"[quant_utils] Calibration batch {batch_idx} processed.")

    # 移除 hooks
    for h in hooks:
        h.remove()

    print(f"[quant_utils] Collected activation stats for {len(act_max)} linear layers.")
    return act_max


def compute_smoothquant_scales(
    model: nn.Module,
    act_max: Dict[str, torch.Tensor],
    alpha: float,
):
    """
    根據 activation max 與 weight max 計算 SmoothQuant 的 scale，
    並直接對權重做縮放。
    回傳 input_scales[name] = 1 / s (給 forward 乘上)。
    """
    input_scales: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(module, CastedLinear):
            continue
        if name not in act_max:
            print(f"[quant_utils][WARN] No activation stats for layer {name}, skip SmoothQuant for this layer.")
            continue

        x_max = act_max[name].to(module.weight.device)  # [C_in]
        # 權重 per-input-channel 絕對值最大值：沿著 output dim 取 max → [C_in]
        w_max = module.weight.detach().abs().amax(dim=0)

        x_max = x_max.clamp(min=1e-5)
        w_max = w_max.clamp(min=1e-5)

        # SmoothQuant: s = X_max^alpha / W_max^(1-alpha)
        s = (x_max.pow(alpha)) / (w_max.pow(1.0 - alpha))
        s = s.clamp(min=1e-3, max=1e3)

        # 權重縮放: W' = W * diag(s)
        with torch.no_grad():
            module.weight.mul_(s)

        input_scales[name] = (1.0 / s).to(torch.float32)
        print(
            f"[quant_utils] SmoothQuant applied on {name}: "
            f"mean(s)={s.mean().item():.4f}, min(s)={s.min().item():.4f}, max(s)={s.max().item():.4f}"
        )

    return input_scales


def _resolve_parent_and_attr(model: nn.Module, full_name: str):
    """
    根據 'inner.L_level.layers.0.self_attn.qkv_proj' 這種 full_name，
    找到 parent module 以及最後一層屬性名稱或索引。
    """
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    attr = parts[-1]
    return parent, attr


def replace_castedlinear_with_w8a8(
    model: nn.Module,
    input_scales: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    將所有 CastedLinear 取代成 W8A8Linear。
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            targets.append((name, module))

    print(f"[quant_utils] Replacing {len(targets)} CastedLinear layers with W8A8Linear...")

    for name, module in targets:
        parent, attr = _resolve_parent_and_attr(model, name)
        scale = None
        if input_scales is not None and name in input_scales:
            scale = input_scales[name]
        new_linear = W8A8Linear(module, input_scale=scale)

        if attr.isdigit():
            idx = int(attr)
            parent[idx] = new_linear
        else:
            setattr(parent, attr, new_linear)

        print(f"[quant_utils] Replaced layer {name} with W8A8Linear.")

    return model


def prepare_model_for_eval(
    model: nn.Module,
    calib_iterator: Iterable,
    device: torch.device,
    use_smoothquant: bool,
    sq_alpha: float,
    sq_max_calib_batches: int,
):
    """
    統一入口：
    - use_smoothquant=False: 直接 Naive W8A8（只做 fake quant，不做 SmoothQuant 權重平滑）
    - use_smoothquant=True: 先跑校正收集 activation 再做 SmoothQuant，最後再換成 W8A8Linear。
    """
    if use_smoothquant:
        print(
            f"[quant_utils] === SmoothQuant W8A8 模型準備開始，alpha={sq_alpha}，"
            f"calib_batches={sq_max_calib_batches} ==="
        )
        act_max = collect_activation_and_weight_stats(
            model=model,
            calib_iterator=calib_iterator,
            device=device,
            max_batches=sq_max_calib_batches,
        )
        input_scales = compute_smoothquant_scales(model, act_max, alpha=sq_alpha)
        replace_castedlinear_with_w8a8(model, input_scales=input_scales)
    else:
        print("[quant_utils] === Naive W8A8 模型準備開始（不使用 SmoothQuant） ===")
        replace_castedlinear_with_w8a8(model, input_scales=None)

    return model
