import os
import sys
import yaml
from dataclasses import is_dataclass, fields
from typing import Any, Dict, Iterable, Tuple

import torch

# --- [FIX] 修正 Import 路徑 ---
# 原本: from trm import ... (錯誤，因為 trm.py 在 models/recursive_reasoning/)
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.losses import IGNORE_LABEL_ID

from utils.quant_utils import prepare_model_for_eval


def _parse_cli_args(argv):
    """
    非 Hydra 版本的簡易參數解析：
    支援 key=value 與類似 Hydra 的 list 字串，如:
      data_paths="[data/sudoku-extreme-1k-aug-1000]"
    """
    parsed: Dict[str, Any] = {}
    for arg in argv:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)

        v = v.strip()
        # bool
        if v.lower() == "true":
            parsed[k] = True
            continue
        if v.lower() == "false":
            parsed[k] = False
            continue

        # list like [a,b] or ["a","b"]
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                parsed[k] = []
                continue
            # split by comma
            items = [x.strip() for x in inner.split(",")]
            # remove quotes
            items = [x.strip("\"'") for x in items if x]
            parsed[k] = items
            continue

        # int / float
        try:
            if "." in v:
                parsed[k] = float(v)
            else:
                parsed[k] = int(v)
            continue
        except ValueError:
            pass

        parsed[k] = v
    return parsed


def _auto_find_all_config(load_checkpoint: str) -> str:
    """
    根據 checkpoint 路徑自動尋找 all_config.yaml：
    1) 同一資料夾下的 all_config.yaml
    2) 專案根目錄的 all_config.yaml
    找不到則噴錯。
    """
    candidates = []
    if load_checkpoint:
        ckpt_dir = os.path.dirname(load_checkpoint)
        candidates.append(os.path.join(ckpt_dir, "all_config.yaml"))
    candidates.append("all_config.yaml")

    for path in candidates:
        if path and os.path.exists(path):
            print(f"[run_eval] Using all_config.yaml at: {path}")
            return path

    raise FileNotFoundError(
        f"Could not find all_config.yaml. Tried: {candidates}. "
        "Please pass model_config_path=/path/to/all_config.yaml in the command."
    )


def _build_dataset_config(parsed_args, all_cfg_dict):
    data_paths = parsed_args.get("data_paths", all_cfg_dict.get("data_paths", []))
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    if not data_paths:
        raise ValueError("data_paths is empty. Please pass data_paths=[...] in command.")

    global_batch_size = int(parsed_args.get("global_batch_size", 128))
    seed = int(parsed_args.get("seed", all_cfg_dict.get("seed", 0)))

    ds_cfg = PuzzleDatasetConfig(
        seed=seed,
        dataset_paths=data_paths,
        global_batch_size=global_batch_size,
        test_set_mode=True,      # 評估模式
        epochs_per_iter=1,       # 評估時不需要多 epoch 疊加
        rank=0,
        num_replicas=1,
    )
    return ds_cfg


def _build_trm_config(parsed_args, all_cfg_dict, dataset_metadata) -> Dict[str, Any]:
    """
    從 all_config.yaml 的 arch 區段 + dataset metadata + CLI 覆寫
    建立 TinyRecursiveReasoningModel_ACTV1Config 所需的 config dict。
    """
    arch_cfg = dict(all_cfg_dict.get("arch", {}))

    # CLI 覆寫 arch.*
    for k, v in parsed_args.items():
        if not k.startswith("arch."):
            continue
        sub_key = k[len("arch."):]
        print(f"[run_eval] Override arch.{sub_key} = {v}")
        arch_cfg[sub_key] = v

    required_keys = [
        "H_cycles", "L_cycles", "H_layers", "L_layers",
        "hidden_size", "expansion", "num_heads",
        "pos_encodings", "halt_max_steps", "halt_exploration_prob",
    ]
    for k in required_keys:
        if k not in arch_cfg:
            raise KeyError(f"Key arch.{k} not found in all_config.yaml")

    # --- [分析確認] ---
    # 這裡回答你的疑問：vocab_size 是從 dataset_metadata 讀取的，
    # 所以 all_config.yaml 裡面沒有 vocab_size 是正常的。
    seq_len = int(dataset_metadata.seq_len)
    vocab_size = int(dataset_metadata.vocab_size)
    num_puzzle_identifiers = int(dataset_metadata.num_puzzle_identifiers)

    # 批次大小使用評估時的 global_batch_size
    batch_size = int(parsed_args.get("global_batch_size", all_cfg_dict.get("global_batch_size", 128)))

    cfg: Dict[str, Any] = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": int(arch_cfg.get("puzzle_emb_ndim", 512)),
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "vocab_size": vocab_size,
        "H_cycles": int(arch_cfg["H_cycles"]),
        "L_cycles": int(arch_cfg["L_cycles"]),
        "H_layers": int(arch_cfg.get("H_layers", 0)),
        "L_layers": int(arch_cfg["L_layers"]),
        "hidden_size": int(arch_cfg["hidden_size"]),
        "expansion": float(arch_cfg["expansion"]),
        "num_heads": int(arch_cfg["num_heads"]),
        "pos_encodings": str(arch_cfg["pos_encodings"]),
        "halt_max_steps": int(arch_cfg["halt_max_steps"]),
        "halt_exploration_prob": float(arch_cfg["halt_exploration_prob"]),
        "forward_dtype": str(arch_cfg.get("forward_dtype", "bfloat16")),
        "mlp_t": bool(arch_cfg.get("mlp_t", False)),
        "puzzle_emb_len": int(arch_cfg.get("puzzle_emb_len", 16)),
        "no_ACT_continue": bool(arch_cfg.get("no_ACT_continue", True)),
    }

    print("[run_eval] TinyRecursiveReasoningModel_ACTV1Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    return cfg


def _to_device(obj, device):
    """遞迴地把 dataclass / dict / tensor 搬到目標 device。"""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if is_dataclass(obj):
        kwargs = {}
        for f in fields(obj):
            kwargs[f.name] = _to_device(getattr(obj, f.name), device)
        return type(obj)(**kwargs)
    return obj


def _load_trm_from_checkpoint(
    ckpt_path: str,
    model_cfg_dict: Dict[str, Any],
    device: torch.device,
) -> TinyRecursiveReasoningModel_ACTV1:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[run_eval] Loading checkpoint from {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        print("[run_eval] Found nested 'state_dict' in checkpoint.")
    else:
        state_dict = raw
        print("[run_eval] Using checkpoint as flat state_dict.")

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("_orig_mod.model."):
            new_k = new_k[len("_orig_mod.model.") :]
        elif new_k.startswith("model."):
            new_k = new_k[len("model.") :]
        cleaned[new_k] = v

    print("[run_eval] Example state_dict keys after cleaning:")
    for i, k in enumerate(cleaned.keys()):
        print(f"  {i:02d}: {k}")
        if i >= 9:
            break

    model = TinyRecursiveReasoningModel_ACTV1(model_cfg_dict)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if missing:
        print(f"[run_eval][WARN] Missing keys when loading state_dict ({len(missing)}):")
        for k in missing:
            print(f"  MISSING: {k}")
    if unexpected:
        print(f"[run_eval][WARN] Unexpected keys when loading state_dict ({len(unexpected)}):")
        for k in unexpected:
            print(f"  UNEXPECTED: {k}")

    model.to(device)
    model.eval()
    return model


def _iterate_dataset(ds: Iterable):
    """
    PuzzleDataset 會回傳 (set_name, batch, effective_bs)，
    也允許未來改動成直接回傳 batch dict 的情況。
    這個小 helper 可以把兩種情況統一處理。
    """
    for sample in ds:
        if isinstance(sample, tuple) and len(sample) == 3:
            set_name, batch, eff_bs = sample
        else:
            set_name, batch, eff_bs = None, sample, None
        yield set_name, batch, eff_bs


def main():
    parsed = _parse_cli_args(sys.argv[1:])
    print("[run_eval] Parsed CLI args:")
    for k, v in parsed.items():
        print(f"  {k} = {v}")

    arch_name = parsed.get("arch", "trm")
    if arch_name != "trm":
        raise ValueError(f"Only arch=trm is supported in this eval script, got: {arch_name}")

    ckpt_path = parsed.get("load_checkpoint", None)
    if not ckpt_path:
        raise ValueError("load_checkpoint is required, e.g. load_checkpoint=./checkpoints/sudoku_att/step_21700")

    model_cfg_path = parsed.get("model_config_path", None)
    if model_cfg_path is None:
        model_cfg_path = _auto_find_all_config(ckpt_path)

    with open(model_cfg_path, "r") as f:
        all_cfg_dict = yaml.safe_load(f)

    # --- 準備 Dataset ---
    ds_cfg = _build_dataset_config(parsed, all_cfg_dict)
    print("[run_eval] PuzzleDatasetConfig:")
    print(ds_cfg)

    # 校正與評估分開建立 Dataset，避免 IterableDataset 被消耗後難以重複使用
    calib_dataset = PuzzleDataset(ds_cfg, split="test")
    eval_dataset = PuzzleDataset(ds_cfg, split="test")

    # 讀取 metadata 以建立 TRM Config
    metadata = calib_dataset.metadata
    print("[run_eval] Dataset metadata:")
    print(f"  seq_len={metadata.seq_len}, vocab_size={metadata.vocab_size}, "
          f"num_puzzle_identifiers={metadata.num_puzzle_identifiers}")

    model_cfg_dict = _build_trm_config(parsed, all_cfg_dict, metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_eval] Using device: {device}")

    # --- 建立並載入模型 ---
    model = _load_trm_from_checkpoint(ckpt_path, model_cfg_dict, device)

    # --- 準備量化 (Naive W8A8 或 SmoothQuant W8A8) ---
    use_smoothquant = bool(parsed.get("use_smoothquant", False))
    sq_alpha = float(parsed.get("sq_alpha", 0.5))
    sq_max_calib_batches = int(parsed.get("sq_max_calib_batches", 32))

    print(f"[run_eval] Quantization settings: use_smoothquant={use_smoothquant}, "
          f"sq_alpha={sq_alpha}, sq_max_calib_batches={sq_max_calib_batches}")

    calibrator = _iterate_dataset(calib_dataset)
    prepare_model_for_eval(
        model=model,
        calib_iterator=calibrator,
        device=device,
        use_smoothquant=use_smoothquant,
        sq_alpha=sq_alpha,
        sq_max_calib_batches=sq_max_calib_batches,
    )

    # --- 評估迴圈 ---
    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    correct_puzzles = 0

    print("[run_eval] Start evaluation...")
    with torch.no_grad():
        for step_idx, (set_name, batch, eff_bs) in enumerate(_iterate_dataset(eval_dataset)):
            batch = {k: v.to(device) for k, v in batch.items()}

            carry = model.initial_carry(batch)
            carry = _to_device(carry, device)

            carry, outputs = model(carry, batch)
            logits = outputs["logits"]  # [B, L, V]
            labels = batch["labels"]    # [B, L]

            preds = logits.argmax(dim=-1)

            # token-level accuracy（忽略 IGNORE_LABEL_ID）
            mask = labels != IGNORE_LABEL_ID
            correct = (preds == labels) & mask
            correct_tokens += correct.sum().item()
            total_tokens += mask.sum().item()

            # puzzle-level accuracy（所有非 ignore label token 都正確才算 1 題）
            per_puzzle_correct = ((preds == labels) | ~mask).all(dim=-1)
            correct_puzzles += per_puzzle_correct.sum().item()
            total_puzzles += per_puzzle_correct.numel()

            if step_idx % 10 == 0:
                tok_acc = correct_tokens / max(total_tokens, 1)
                puzz_acc = correct_puzzles / max(total_puzzles, 1)
                print(f"[run_eval] step={step_idx}, token_acc={tok_acc:.4f}, puzzle_acc={puzz_acc:.4f}")

    final_token_acc = correct_tokens / max(total_tokens, 1)
    final_puzzle_acc = correct_puzzles / max(total_puzzles, 1)
    print(f"[run_eval] Final token-level accuracy:  {final_token_acc:.4f}")
    print(f"[run_eval] Final puzzle-level accuracy: {final_puzzle_acc:.4f}")

    # 如果有指定 checkpoint_path，將結果寫入檔案
    out_dir = parsed.get("checkpoint_path", None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "eval_results.txt")
        with open(out_path, "w") as f:
            f.write(f"token_acc={final_token_acc:.6f}\n")
            f.write(f"puzzle_acc={final_puzzle_acc:.6f}\n")
        print(f"[run_eval] Saved results to {out_path}")


if __name__ == "__main__":
    main()
