# utils/quant_utils.py
import torch
import torch.nn as nn
from typing import Dict, List, Any

from models.layers import CastedLinear


class W8A8Linear(nn.Module):
    """
    模擬 W8A8 量化的 Linear 層。
    - 權重：per-output-channel fake quant (對稱)
    - Activation：per-tensor fake quant (對稱)
    - 支援 SmoothQuant 的 input_scale（s^-1）
    """
    def __init__(self, original_layer: CastedLinear, input_scale: torch.Tensor | None = None):
        super().__init__()

        weight = original_layer.weight.detach().clone()
        self.weight = nn.Parameter(weight)

        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

        in_features = weight.shape[1]
        if input_scale is not None:
            input_scale = input_scale.detach().clone()
        else:
            input_scale = torch.ones(in_features, dtype=weight.dtype)

        self.register_buffer("input_scale", input_scale)

    @staticmethod
    def _fake_quant_tensor(
        t: torch.Tensor, num_bits: int = 8, per_channel: bool = False, ch_axis: int = 1
    ) -> torch.Tensor:
        """
        對稱假量化：
        - per_channel=True 時，用每個 channel 的 max 做縮放。
        - per_channel=False 時，整個 tensor 共用一組 scale。
        """
        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        # 統一用 float32 做 quant / dequant 計算，比較穩定
        orig_dtype = t.dtype
        x = t.to(torch.float32)

        if per_channel:
            # 假設權重 shape [out, in]，以 in 維度為 channel（ch_axis=1）
            abs_max = x.abs().amax(dim=ch_axis, keepdim=True).clamp(min=1e-5)
            scale = abs_max / qmax
            x_q = torch.round(x / scale).clamp(qmin, qmax)
            x_deq = x_q * scale
        else:
            abs_max = x.abs().amax().clamp(min=1e-5)
            scale = abs_max / qmax
            x_q = torch.round(x / scale).clamp(qmin, qmax)
            x_deq = x_q * scale

        return x_deq.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SmoothQuant input scaling：X' = X * s^-1
        x = x * self.input_scale.to(x.device).to(x.dtype)

        # A8：per-tensor 假量化
        x_q = self._fake_quant_tensor(x, num_bits=8, per_channel=False)

        # W8：per-output-channel 假量化
        w_q = self._fake_quant_tensor(self.weight, num_bits=8, per_channel=True, ch_axis=1)

        return nn.functional.linear(x_q, w_q, self.bias)


# -----------------------------
# Calibration：收集 Activation 統計
# -----------------------------
def collect_calib_batches(dataloader, max_batches: int) -> List[Dict[str, torch.Tensor]]:
    """
    從 (set_name, batch, global_batch_size) 的 dataloader 中擷取前 max_batches 個 batch（只留 batch dict）。
    """
    calib_batches: List[Dict[str, torch.Tensor]] = []
    for i, (set_name, batch, global_bs) in enumerate(dataloader):
        calib_batches.append(batch)
        if i + 1 >= max_batches:
            break
    print(f"[quant_utils] Collected {len(calib_batches)} calibration batches.")
    return calib_batches


@torch.no_grad()
def calibrate_model(
    model: nn.Module,
    calib_batches: List[Dict[str, torch.Tensor]],
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    在 calibration batches 上收集各 CastedLinear 的 activation max（per input channel）。
    回傳：{module_full_name: act_max_vector}
    """
    model.eval()
    act_abs_max: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(module, inputs, output):
            # inputs[0] shape: [B, L, D] 或 [B, D]
            x = inputs[0].detach()
            # 取 batch / seq 維度上的 max，留下最後一維 channel
            if x.dim() == 3:
                cur_max = x.abs().amax(dim=(0, 1)).cpu()
            elif x.dim() == 2:
                cur_max = x.abs().amax(dim=0).cpu()
            else:
                cur_max = x.abs().amax().view(-1).cpu()

            if name in act_abs_max:
                act_abs_max[name] = torch.maximum(act_abs_max[name], cur_max)
            else:
                act_abs_max[name] = cur_max
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            print(f"[quant_utils] Register hook on CastedLinear: {name}")
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # 跑 calibration
    for i, batch in enumerate(calib_batches):
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        carry = model.initial_carry(batch_dev)
        _carry, _outputs = model(carry, batch_dev)

        if i % 5 == 0:
            print(f"[quant_utils] Calibration batch {i} processed.")

    for h in hooks:
        h.remove()

    print(f"[quant_utils] Collected activation stats for {len(act_abs_max)} linear layers.")
    return act_abs_max


# -----------------------------
# SmoothQuant：重寫權重 + input_scale
# -----------------------------
@torch.no_grad()
def apply_smoothquant(
    model: nn.Module,
    act_abs_max: Dict[str, torch.Tensor],
    alpha: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    對每個 CastedLinear 套用 SmoothQuant：
      s = X_max^alpha / W_max^(1-alpha)
      W' = W * diag(s)
    並回傳 input_scale_map：layer_name -> (1/s)
    """
    input_scales_map: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(module, CastedLinear):
            continue
        if name not in act_abs_max:
            continue

        x_max = act_abs_max[name].to(module.weight.device)
        w_max = module.weight.detach().abs().amax(dim=0)

        x_max = x_max.clamp(min=1e-5)
        w_max = w_max.clamp(min=1e-5)

        # s = X_max^alpha / W_max^(1-alpha)
        s = x_max.pow(alpha) / w_max.pow(1.0 - alpha)

        # 重寫權重：W' = W * diag(s)
        module.weight.mul_(s.view(1, -1))

        input_scales_map[name] = (1.0 / s).to(module.weight.device)

        print(
            f"[quant_utils] SmoothQuant applied on {name}: "
            f"mean(s)={s.mean().item():.4f}, "
            f"min(s)={s.min().item():.4f}, "
            f"max(s)={s.max().item():.4f}"
        )

    return input_scales_map


# -----------------------------
# 量化層替換：CastedLinear -> W8A8Linear
# -----------------------------
def replace_with_quantized_layers(
    model: nn.Module,
    input_scales_map: Dict[str, torch.Tensor] | None = None,
):
    """
    將模型之中所有 CastedLinear 換成 W8A8Linear。
    - 若 input_scales_map 有對應 layer 名稱，則使用 SmoothQuant 的 input_scale（s^-1）。
    - 若沒有（或整體為 None），則 input_scale=1，代表 Naive W8A8。
    """
    # 先蒐集要替換的 module
    to_replace: List[tuple[str, nn.Module, torch.Tensor | None]] = []
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            scale = None
            if input_scales_map is not None and name in input_scales_map:
                scale = input_scales_map[name]
            to_replace.append((name, module, scale))

    print(f"[quant_utils] Replacing {len(to_replace)} CastedLinear layers with W8A8Linear...")

    for full_name, module, scale in to_replace:
        # 拆 parent / child 名稱
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = full_name

        q_layer = W8A8Linear(module, input_scale=scale)
        setattr(parent, child_name, q_layer)
        print(f"[quant_utils] Replaced layer {full_name} with W8A8Linear.")


