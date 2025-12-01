# utils/quant_utils.py
import torch
import torch.nn as nn
from typing import Dict, List, Any
import yaml

from collections import defaultdict
from models.layers import CastedLinear

def _move_to_device(obj: Any, device: str):
    """遞迴將 dataclass / dict / tensor 移到指定 device。"""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    # TRM 的 carry 是 dataclass，有 __dataclass_fields__
    if hasattr(obj, "__dataclass_fields__"):
        for f in obj.__dataclass_fields__:
            setattr(obj, f, _move_to_device(getattr(obj, f), device))
        return obj
    return obj

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
        # [1] SmoothQuant Input Scaling
        scale = self.input_scale.to(x.device).to(x.dtype)
        view_shape = [1] * (x.dim() - 1) + [scale.shape[-1]]
        x = x * scale.view(*view_shape)

        # [2] Activation Fake Quant
        x_q = self._fake_quant_tensor(x, num_bits=8, per_channel=False)

        # [3] Weight Fake Quant
        w_q = self._fake_quant_tensor(self.weight, num_bits=8, per_channel=True, ch_axis=1)

        # [4] 統一 Dtype (關鍵步驟)
        # 將權重轉為與輸入相同的型別 (例如 BFloat16)
        common_dtype = x_q.dtype
        w_q = w_q.to(common_dtype)
        
        # 處理 Bias 的 Dtype
        if self.bias is not None:
            bias = self.bias.to(common_dtype)
        else:
            bias = None

        # [5] Linear 運算 (使用局部變數 bias)
        return nn.functional.linear(x_q, w_q, bias)

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
        carry = _move_to_device(carry, device)
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
    alpha: float = 0.5,
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


def export_int8_config(
    model,
    act_abs_max: dict,
    input_scales_map: dict,
    bits: int,
    yaml_path: str,
    weight_path: str,
):
    """
    將目前 (已套用 SmoothQuant 權重) 的模型，導出成 INT8 量化設定：
      - sudoku_trm_int8_config.yaml: 每層的 scale / shape / 有無 bias
      - sudoku_trm_int8_weights.pt: 量化後的 weight_int8 與 bias_int32

    參數：
      model            : 已經執行完 apply_smoothquant() 的 TRM 模型（仍為 CastedLinear，尚未換成 W8A8Linear）
      act_abs_max      : calibrate_model() 回傳的 activation max dict {layer_name: tensor[in_features]}
      input_scales_map : apply_smoothquant() 回傳的 SmoothQuant input_scale dict {layer_name: tensor[in_features]}
      bits             : 目前使用 8  bits
      yaml_path        : YAML 輸出路徑
      weight_path      : .pt 權重輸出路徑
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    layers_cfg = {}
    weight_tensors = {}

    for name, module in model.named_modules():
        if not isinstance(module, CastedLinear):
            continue

        W = module.weight.detach().to(torch.float32)
        out_features, in_features = W.shape

        layer_info = {
            "bits": int(bits),
            "in_features": int(in_features),
            "out_features": int(out_features),
        }

        # 1) SmoothQuant input_scale（X' = X * input_scale）
        if name in input_scales_map:
            input_scale = input_scales_map[name].detach().to(torch.float32)
        else:
            input_scale = torch.ones(in_features, dtype=torch.float32)

        layer_info["input_scale"] = input_scale.cpu().tolist()

        # 2) Activation scale (act_scale)
        #    我們先把 SmoothQuant 的 input_scale 也考慮進去：
        #    X_max 來自 calibrate_model（原始 X），X'_max = X_max * input_scale
        if name in act_abs_max:
            X_max = act_abs_max[name].detach().to(torch.float32)  # [in_features]
            input_scale=input_scale.to(X_max.device)             # 為了和 X_max 相乘，統一搬到 X_max.device（CPU）
            X_prime_max = X_max * input_scale                    # [in_features]
            A_max = X_prime_max.abs().max().clamp(min=1e-8)
            act_scale = (A_max / qmax).item()
        else:
            act_scale = 1.0

        layer_info["act_scale"] = float(act_scale)

        # 3) Weight scale（per-output-channel）
        w_max = W.abs().amax(dim=1)          # [out_features]
        w_max = w_max.clamp(min=1e-8)
        w_scale = w_max / qmax              # [out_features]
        layer_info["weight_scale"] = w_scale.cpu().tolist()

        # 4) 量化 weight → int8
        w_int8 = torch.round(W / w_scale.view(-1, 1)).clamp(qmin, qmax).to(torch.int8)

        # 5) Output scale = act_scale * weight_scale（per-output-channel）
        #    之後 int32 accumulator * output_scale[o] = approximate FP32 output
        output_scale = act_scale * w_scale  # [out_features]
        layer_info["output_scale"] = output_scale.cpu().tolist()

        # 6) Bias 量化成 int32
        if module.bias is not None:
            b = module.bias.detach().to(torch.float32)  # [out_features]
            # y_fp ≈ (acc_int32 + b_int32) * output_scale
            # => b_int32 ≈ b_fp / output_scale
            b_int32 = torch.round(b / output_scale).to(torch.int32)
            has_bias = True
        else:
            b_int32 = torch.zeros(out_features, dtype=torch.int32)
            has_bias = False

        layer_info["has_bias"] = bool(has_bias)

        # 7) 收集權重／bias tensor（存 .pt）
        weight_tensors[f"{name}.weight_int8"] = w_int8.cpu()
        weight_tensors[f"{name}.bias_int32"] = b_int32.cpu()

        # 8) 存入 layers config
        layers_cfg[name] = layer_info

    full_cfg = {
        "bits": int(bits),
        "layers": layers_cfg,
    }

    # 寫 YAML
    with open(yaml_path, "w") as f:
        yaml.safe_dump(full_cfg, f, sort_keys=True)

    # 寫 quantized weights
    torch.save(weight_tensors, weight_path)

    print(f"[quant_utils] Saved INT8 quant config to {yaml_path}")
    print(f"[quant_utils] Saved INT8 quant weights to {weight_path}")


# =============================
# Bit-exact INT8 Linear 模擬
# =============================
class Int8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_int8: torch.Tensor,
        bias_int32: torch.Tensor,
        input_scale: torch.Tensor,
        act_scale: float,
        output_scale: torch.Tensor,
        bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        # 量化後的整數權重 / bias
        self.register_buffer("weight_int8", weight_int8.to(torch.int8))
        if bias_int32 is not None:
            self.register_buffer("bias_int32", bias_int32.to(torch.int32))
        else:
            self.bias_int32 = None

        # SmoothQuant 的輸入 scaling（per-channel）
        self.register_buffer("input_scale", input_scale.to(torch.float32))

        # activation per-tensor scale
        self.act_scale = float(act_scale)

        # output_scale 可能是 scalar 或 per-channel，統一成 float32 buffer
        self.register_buffer("output_scale", output_scale.to(torch.float32))

    @staticmethod
    def _quantize_to_int8(x: torch.Tensor, act_scale: float, bits: int = 8) -> torch.Tensor:
        qmax = 2 ** (bits - 1) - 1  # 127
        qmin = -2 ** (bits - 1)     # -128

        if act_scale <= 0:
            raise ValueError(f"act_scale must be > 0, got {act_scale}")

        x_scaled = x / act_scale
        x_rounded = torch.round(x_scaled)
        x_clamped = torch.clamp(x_rounded, qmin, qmax)
        return x_clamped.to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        整數路徑（在 CUDA 上用 FP32 精確模擬）：
          1) x_fp32 = x * input_scale
          2) x_int8 = quantize(x_fp32 / act_scale)
          3) acc = (x_int8 @ w_int8^T)  # 在 FP32 中做，數值等同 int32 累加
          4) acc += bias_int32
          5) y = acc * output_scale
        """

        # 1. 先轉成 float32，乘 SmoothQuant input_scale
        x_fp32 = x.to(torch.float32)
        inp_scale = self.input_scale.to(x_fp32.device, x_fp32.dtype)  # [in_features]
        x_scaled = x_fp32 * inp_scale  # broadcasting on last dim

        # 2. 固定 act_scale 做 per-tensor quant → int8
        x_int8 = self._quantize_to_int8(x_scaled, self.act_scale, self.bits)  # int8
        w_int8 = self.weight_int8.to(x_int8.device)                           # int8

        # 3. 用 FP32 模擬 int8×int8→int32（不會有數值誤差）
        x_int = x_int8.to(torch.float32)
        w_int = w_int8.to(torch.float32)
        acc = x_int @ w_int.t()  # [N, out_features], float32，數值上是整數

        # 4. 加上 bias（原本設計為 int32 累加後再加）
        if self.bias_int32 is not None:
            acc = acc + self.bias_int32.to(acc.device).to(torch.float32)

        # 5. 乘上 output_scale 回到 float
        out_scale = self.output_scale.to(acc.device, acc.dtype)
        y = acc * out_scale  # float32

        # 視需求決定回傳型別：這裡回傳 float32；若希望跟原層 dtype 一致可改成 y.to(x.dtype)
        return y

# class Int8Linear(nn.Module):
#     """
#     Bit-exact INT8 Linear 模擬：

#     forward(x_fp):
#       1) z = x_fp * input_scale           # SmoothQuant pre-scale
#       2) x_int8 = round(z / act_scale)    # clamp 到 [-128,127]
#       3) acc_int32 = x_int8 @ weight_int8^T + bias_int32
#       4) y_fp = acc_int32 * output_scale  # per-output-channel
#       5) cast 回原本 dtype (bf16 / fp32)

#     所有參數來自 export_int8_config 輸出的 YAML + .pt：
#       - input_scale: [in_features]
#       - act_scale:   scalar
#       - weight_int8: [out_features, in_features]
#       - bias_int32:  [out_features]
#       - output_scale:[out_features]
#     """

#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         weight_int8: torch.Tensor,   # [out, in], int8
#         bias_int32: torch.Tensor,    # [out], int32
#         input_scale: torch.Tensor,   # [in], float32
#         act_scale: float,            # scalar
#         output_scale: torch.Tensor,  # [out], float32
#         bits: int = 8,
#     ):
#         super().__init__()
#         self.in_features = int(in_features)
#         self.out_features = int(out_features)
#         self.bits = int(bits)

#         # 量化後權重 / bias / scale 全放 buffer，方便 .to(device)
#         self.register_buffer("weight_int8", weight_int8.to(torch.int8))
#         self.register_buffer("bias_int32", bias_int32.to(torch.int32))
#         self.register_buffer("input_scale", input_scale.to(torch.float32))
#         self.register_buffer("output_scale", output_scale.to(torch.float32))

#         # act_scale 用 float 儲存即可
#         self.act_scale = float(act_scale)

#     def forward(self, x_fp: torch.Tensor) -> torch.Tensor:
#         """
#         x_fp: [..., in_features]，可以是 [B, S, H] 或 [B, H]
#         回傳: [..., out_features]，dtype 與 x_fp 相同
#         """
#         orig_dtype = x_fp.dtype
#         device = x_fp.device

#         x = x_fp.to(torch.float32)
#         last_dim = x.shape[-1]
#         assert (
#             last_dim == self.in_features
#         ), f"Int8Linear: in_features mismatch, got {last_dim}, expected {self.in_features}"

#         # 攤平成 [N, in]
#         x_flat = x.view(-1, self.in_features)  # [N, in]

#         # 1) SmoothQuant pre-scale：z = x * input_scale
#         # input_scale: [in] → broadcast 成 [N, in]
#         z = x_flat * self.input_scale.to(device)  # [N, in]

#         # 2) Activation quantization: z → x_int8
#         qmin = -(2 ** (self.bits - 1))
#         qmax = (2 ** (self.bits - 1)) - 1

#         x_int32 = torch.round(z / self.act_scale).clamp(qmin, qmax).to(torch.int32)  # [N, in]

#         # 3) Weight int8 → int32，做 matmul
#         w_int32 = self.weight_int8.to(torch.int32)  # [out, in]
#         acc = x_int32 @ w_int32.t()                 # [N, out], int32

#         # 4) 加上 bias_int32
#         if self.bias_int32 is not None:
#             acc = acc + self.bias_int32.to(device)  # broadcast [out]

#         # 5) Dequant: y_fp32 = acc_int32 * output_scale（per-output-channel）
#         y_fp32 = acc.to(torch.float32) * self.output_scale.to(device).view(1, -1)  # [N, out]

#         # 6) reshape 回原本 batch/seq 維度，並轉回原本 dtype
#         new_shape = x_fp.shape[:-1] + (self.out_features,)
#         y_fp32 = y_fp32.view(*new_shape)

#         return y_fp32.to(orig_dtype)

def build_int8_linear_from_config(
    layer_name: str,
    yaml_path: str,
    weight_path: str,
    device: str = "cuda",
) -> Int8Linear:
    """
    從 sudoku_trm_int8_config_alpha0.5.yaml 與 sudoku_trm_int8_weights_alpha0.5.pt
    建立對應某一層的 Int8Linear。

    用法示例：
      int8_layer = build_int8_linear_from_config(
          layer_name="inner.L_level.layers.0.mlp.down_proj",
          yaml_path="./eval_results/int8_alpha05/sudoku_trm_int8_config_alpha0.5.yaml",
          weight_path="./eval_results/int8_alpha05/sudoku_trm_int8_weights_alpha0.5.pt",
      )
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    global_bits = int(cfg.get("bits", 8))
    layer_cfg = cfg["layers"][layer_name]

    in_features = int(layer_cfg["in_features"])
    out_features = int(layer_cfg["out_features"])
    input_scale = torch.tensor(layer_cfg["input_scale"], dtype=torch.float32)
    act_scale = float(layer_cfg["act_scale"])
    output_scale = torch.tensor(layer_cfg["output_scale"], dtype=torch.float32)

    weight_tensors = torch.load(weight_path, map_location="cpu")
    weight_int8 = weight_tensors[f"{layer_name}.weight_int8"]
    bias_int32 = weight_tensors[f"{layer_name}.bias_int32"]

    bits = int(layer_cfg.get("bits", global_bits))

    int8_layer = Int8Linear(
        in_features=in_features,
        out_features=out_features,
        weight_int8=weight_int8,
        bias_int32=bias_int32,
        input_scale=input_scale,
        act_scale=act_scale,
        output_scale=output_scale,
        bits=bits,
    )

    return int8_layer.to(device)



def replace_with_int8_from_config(
    model: torch.nn.Module,
    yaml_path: str,
    weight_path: str,
    device: str = "cuda",
    target_layers: list[str] | None = None,
):
    """
    將模型中的 CastedLinear 部分或全部替換為 Int8Linear。

    - yaml_path / weight_path: 來自 export_int8_config 的輸出
    - target_layers:
        * None: 替換所有在 YAML config 中有紀錄的層
        * list[str]: 只替換指定的 layer_name
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    layers_cfg = cfg["layers"]
    weight_tensors = torch.load(weight_path, map_location="cpu")

    if target_layers is None:
        target_layers = list(layers_cfg.keys())
    else:
        # 避免打錯名字
        target_layers = [name for name in target_layers if name in layers_cfg]

    # 先收集所有需要替換的 (full_name, parent_module, child_name)
    modules_to_replace = []

    for full_name, module in model.named_modules():
        if not isinstance(module, CastedLinear):
            continue
        if full_name not in target_layers:
            continue

        # 找 parent module 與 child 名稱
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = full_name

        modules_to_replace.append((full_name, parent, child_name))

    # 實際替換
    replaced = 0
    for full_name, parent, child_name in modules_to_replace:
        layer_cfg = layers_cfg[full_name]
        in_features = int(layer_cfg["in_features"])
        out_features = int(layer_cfg["out_features"])
        input_scale = torch.tensor(layer_cfg["input_scale"], dtype=torch.float32)
        act_scale = float(layer_cfg["act_scale"])
        output_scale = torch.tensor(layer_cfg["output_scale"], dtype=torch.float32)

        w_int8 = weight_tensors[f"{full_name}.weight_int8"]
        b_int32 = weight_tensors[f"{full_name}.bias_int32"]

        bits = int(layer_cfg.get("bits", cfg.get("bits", 8)))

        int8_layer = Int8Linear(
            in_features=in_features,
            out_features=out_features,
            weight_int8=w_int8,
            bias_int32=b_int32,
            input_scale=input_scale,
            act_scale=act_scale,
            output_scale=output_scale,
            bits=bits,
        ).to(device)

        setattr(parent, child_name, int8_layer)
        replaced += 1
        print(f"[int8_replace] Replaced {full_name} with Int8Linear.")

    print(f"[int8_replace] Total replaced layers: {replaced}")
