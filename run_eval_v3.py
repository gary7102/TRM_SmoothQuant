import os
import sys
from typing import Any, Dict, List, Optional

import torch

# 從既有的 run_eval_v2 匯入共用邏輯（資料載入、TRM 組態、ACT 評估）
from run_eval_v2 import (
    build_dataloaders,
    load_all_config,
    build_trm_config,
    load_trm_from_checkpoint,
    evaluate_trm,
)

# 從 quant_utils 匯入「整數路徑」替換函式
from utils.quant_utils import replace_with_int8_from_config


# ---------------------------------------------------------
# 1. CLI 解析（簡化版，與 run_eval_v2 類似）
# ---------------------------------------------------------
def parse_cli_args() -> Dict[str, Any]:
    """
    解析命令列參數，格式：
      key=value
    其中：
      - 若 value 是 "[a,b,c]" 會被視為 list[str]
      - 'true'/'false' 會轉為 bool
      - 其他嘗試轉成 int / float，失敗則保留字串
    """
    cli_args: Dict[str, Any] = {}

    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        k = k.strip()
        v = v.strip()

        # list 形式: [a,b,c]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if inner:
                items: List[str] = []
                for item in inner.split(","):
                    s = item.strip()
                    # 去掉兩邊引號
                    if (s.startswith("'") and s.endswith("'")) or (
                        s.startswith('"') and s.endswith('"')
                    ):
                        s = s[1:-1]
                    items.append(s)
                v_converted: Any = items
            else:
                v_converted = []
        else:
            lower_v = v.lower()
            if lower_v == "true":
                v_converted = True
            elif lower_v == "false":
                v_converted = False
            else:
                # 嘗試轉成 int / float
                try:
                    if "." in v:
                        v_converted = float(v)
                    else:
                        v_converted = int(v)
                except ValueError:
                    v_converted = v  # 保留字串

        cli_args[k] = v_converted

    print("[run_eval_int8_partial] Parsed CLI args:")
    for k, v in cli_args.items():
        print(f"  {k} = {v}")
    return cli_args


# ---------------------------------------------------------
# 2. main：載入 TRM、局部 Int8 替換、ACT 評估
# ---------------------------------------------------------
def main() -> None:
    cli_args = parse_cli_args()

    # --- 基本必填參數檢查 ---
    if "data_paths" not in cli_args:
        raise ValueError(
            'Missing "data_paths". Example: data_paths="[data/sudoku-extreme-1k-aug-1000]"'
        )
    if "load_checkpoint" not in cli_args:
        raise ValueError(
            'Missing "load_checkpoint". Example: load_checkpoint=./checkpoints/sudoku_att/step_21700'
        )
    if "int8_config" not in cli_args:
        raise ValueError(
            'Missing "int8_config". Example: int8_config=./eval_results/.../sudoku_trm_int8_config_alpha0.5.yaml'
        )
    if "int8_weight_path" not in cli_args:
        raise ValueError(
            'Missing "int8_weight_path". Example: int8_weight_path=./eval_results/.../sudoku_trm_int8_weights_alpha0.5.pt'
        )

    # data_paths 可能是 list 或單一字串
    data_paths = cli_args["data_paths"]
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    load_checkpoint = str(cli_args["load_checkpoint"])
    checkpoint_path = str(cli_args.get("checkpoint_path", "./eval_results/int8_partial"))
    global_batch_size = int(cli_args.get("global_batch_size", 128))

    int8_config_path = str(cli_args["int8_config"])
    int8_weight_path = str(cli_args["int8_weight_path"])

    # 目標層（可選）：例如 "[inner.q_head]" 或 "[inner.q_head, inner.lm_head]"
    target_layers_raw = cli_args.get("int8_targets", None)
    target_layers: Optional[List[str]] = None
    if target_layers_raw is not None:
        if isinstance(target_layers_raw, list):
            target_layers = [str(x) for x in target_layers_raw]
        elif isinstance(target_layers_raw, str):
            s = target_layers_raw.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            target_layers = [item.strip() for item in s.split(",") if item.strip()]
        else:
            target_layers = [str(target_layers_raw)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_eval_int8_partial] Using device: {device}")

    # --- 讀 all_config.yaml ---
    all_cfg = load_all_config(load_checkpoint)
    seed = all_cfg.get("seed", 0)

    # --- Dataset & Dataloaders（沿用官方 TRM 流程） ---
    dataset, eval_loader, _calib_loader = build_dataloaders(
        data_paths=data_paths,
        global_batch_size=global_batch_size,
        seed=seed,
    )

    # --- 建立 TRM Config ---
    model_config = build_trm_config(
        all_cfg=all_cfg,
        dataset=dataset,
        cli_args=cli_args,  # 支援 arch.L_layers 等 override
        global_batch_size=global_batch_size,
    )

    # --- 載入 TRM + checkpoint（bf16 浮點 baseline） ---
    model = load_trm_from_checkpoint(
        model_config=model_config,
        load_checkpoint=load_checkpoint,
        device=device,
    )

    # -------------------------------------------------
    # 局部 Int8 替換：使用導出的 int8_config + int8_weight_path
    # -------------------------------------------------
    print("[run_eval_int8_partial] Replacing selected layers with Int8Linear...")
    # 這裡假設 quant_utils.replace_with_int8_from_config 的介面為：
    #   replace_with_int8_from_config(model, config_path, weights_path, target_layers=None)
    replace_with_int8_from_config(
        model,
        int8_config_path,
        int8_weight_path,
        device,
        target_layers,
    )

    model.to(device)
    model.eval()

    # --- ACT 多步遞迴推理（沿用 evaluate_trm） ---
    max_steps = model_config.halt_max_steps
    results = evaluate_trm(
        model=model,
        eval_loader=eval_loader,
        device=device,
        max_reasoning_steps=max_steps,
    )

    # --- 儲存結果 ---
    if checkpoint_path:
        os.makedirs(checkpoint_path, exist_ok=True)
        out_path = os.path.join(checkpoint_path, "eval_results_int8_partial.txt")
        with open(out_path, "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"[run_eval_int8_partial] Saved results to {out_path}")


if __name__ == "__main__":
    main()
