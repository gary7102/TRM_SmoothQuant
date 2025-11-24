# run_eval.py
import os
import sys
import yaml
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from pydantic import BaseModel

# 本專案內部模組
# from trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.quant_utils import (
    collect_calib_batches,
    calibrate_model,
    apply_smoothquant,
    replace_with_quantized_layers,
)

IGNORE_LABEL_ID = -100  # 與 models.losses / trm 中一致


# -----------------------------
# 1. 解析 CLI 參數（key=value 形式）
# -----------------------------
def parse_cli_args() -> Dict[str, Any]:
    cli_args: Dict[str, Any] = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        v = v.strip()
        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        elif v.startswith("[") and v.endswith("]"):
            # 解析 list 字串，例如 "[data/sudoku-extreme-1k-aug-1000]"
            vv = v.strip("[]").replace("'", "").replace('"', "")
            v = [x.strip() for x in vv.split(",") if x.strip()]
        else:
            # 嘗試轉成 int / float
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
        cli_args[k] = v

    print("[run_eval] Parsed CLI args:")
    for k, v in cli_args.items():
        print(f"  {k} = {v}")
    return cli_args


class EvalCLIConfig(BaseModel):
    # 基本設定
    arch: str = "trm"
    data_paths: List[str]
    load_checkpoint: str
    checkpoint_path: str
    global_batch_size: int = 128

    # 量化相關
    use_smoothquant: bool = False
    sq_alpha: float = 0.9
    sq_max_calib_batches: int = 128

    class Config:
        extra = "allow"  # 支援 arch.L_layers 這類額外欄位


# -----------------------------
# 2. 建立 Dataset / Dataloader
# -----------------------------
def build_dataloaders(
    data_paths: List[str],
    global_batch_size: int,
    seed: int = 0,
) -> Tuple[PuzzleDataset, DataLoader, DataLoader]:
    """
    建立一個 eval_loader + 一個 calib_loader（避免被耗盡）
    """
    dataset_config = PuzzleDatasetConfig(
        seed=seed,
        dataset_paths=data_paths,
        global_batch_size=global_batch_size,
        test_set_mode=True,   # 評估模式
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    print("[run_eval] PuzzleDatasetConfig:")
    print(f"  seed={dataset_config.seed} "
          f"dataset_paths={dataset_config.dataset_paths} "
          f"global_batch_size={dataset_config.global_batch_size} "
          f"test_set_mode={dataset_config.test_set_mode} "
          f"epochs_per_iter={dataset_config.epochs_per_iter} "
          f"rank={dataset_config.rank} num_replicas={dataset_config.num_replicas}")

    dataset = PuzzleDataset(dataset_config, split="test")
    metadata = dataset.metadata
    print("[run_eval] Dataset metadata:")
    print(f"  seq_len={metadata.seq_len}, vocab_size={metadata.vocab_size}, "
          f"num_puzzle_identifiers={metadata.num_puzzle_identifiers}")

    # DataLoader：一個用於 calibration，一個用於正式 evaluation
    common_kwargs = dict(
        batch_size=None,          # IterableDataset 自己產 batch
        num_workers=0,
        pin_memory=True,
    )

    eval_loader = DataLoader(dataset, **common_kwargs)
    calib_loader = DataLoader(dataset, **common_kwargs)

    return dataset, eval_loader, calib_loader


# -----------------------------
# 3. 建立並載入 TRM 模型
# -----------------------------
def load_all_config(load_checkpoint: str) -> Dict[str, Any]:
    ckpt_dir = os.path.dirname(load_checkpoint)
    all_cfg_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(all_cfg_path):
        raise FileNotFoundError(f"all_config.yaml not found near checkpoint: {all_cfg_path}")
    print(f"[run_eval] Using all_config.yaml at: {all_cfg_path}")
    with open(all_cfg_path, "r") as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg


def build_trm_config(
    all_cfg: Dict[str, Any],
    dataset,
    cli_args: Dict[str, Any],
    global_batch_size: int,
) -> TinyRecursiveReasoningModel_ACTV1Config:
    arch_cfg = dict(all_cfg.get("arch", {}))

    # Dataset metadata
    meta = dataset.metadata
    cfg_dict: Dict[str, Any] = dict(arch_cfg)

    # 由 dataset / CLI 補齊必需欄位
    cfg_dict["batch_size"] = global_batch_size
    cfg_dict["seq_len"] = meta.seq_len
    cfg_dict["vocab_size"] = meta.vocab_size
    cfg_dict["num_puzzle_identifiers"] = meta.num_puzzle_identifiers

    # 若未指定 puzzle_emb_ndim，預設等於 hidden_size
    if "puzzle_emb_ndim" not in cfg_dict:
        cfg_dict["puzzle_emb_ndim"] = cfg_dict.get("hidden_size", 512)

    # 若未指定 puzzle_emb_len / mlp_t 等，套用預設
    cfg_dict.setdefault("puzzle_emb_len", 16)
    cfg_dict.setdefault("mlp_t", False)

    # CLI override: arch.xxx
    for k, v in cli_args.items():
        if k.startswith("arch."):
            key = k.split(".", 1)[1]
            print(f"[run_eval] Override arch.{key} = {v}")
            cfg_dict[key] = v

    # 建立 Pydantic Config
    model_config = TinyRecursiveReasoningModel_ACTV1Config(**cfg_dict)

    print("[run_eval] TinyRecursiveReasoningModel_ACTV1Config:")
    for field, value in model_config.__dict__.items():
        print(f"  {field}: {value}")

    return model_config


def load_trm_from_checkpoint(
    model_config: TinyRecursiveReasoningModel_ACTV1Config,
    load_checkpoint: str,
    device: str,
) -> TinyRecursiveReasoningModel_ACTV1:
    model = TinyRecursiveReasoningModel_ACTV1(model_config)
    model.to(device)

    print(f"[run_eval] Loading checkpoint from {load_checkpoint}")
    raw_ckpt = torch.load(load_checkpoint, map_location="cpu")

    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
        print("[run_eval] Using checkpoint['state_dict']")
    else:
        state_dict = raw_ckpt
        print("[run_eval] Using checkpoint as flat state_dict.")

    cleaned_state = {}
    for k, v in state_dict.items():
        # 去掉 _orig_mod.model. / model. 前綴
        new_k = k
        if new_k.startswith("_orig_mod.model."):
            new_k = new_k[len("_orig_mod.model."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        # 以 TRM_SmoothQuant 推論版本為準：state_dict key 是 inner.*
        cleaned_state[new_k] = v

    # 若 key 沒有 inner. 前綴，就補上
    if not any(k.startswith("inner.") for k in cleaned_state.keys()):
        fixed_state = {}
        for k, v in cleaned_state.items():
            if not k.startswith("inner."):
                fixed_state[f"inner.{k}"] = v
            else:
                fixed_state[k] = v
        cleaned_state = fixed_state

    print("[run_eval] Example state_dict keys after cleaning:")
    for i, k in enumerate(list(cleaned_state.keys())[:10]):
        print(f"  {i:02d}: {k}")

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[run_eval] Missing keys ({len(missing)}): {list(missing)[:10]} ...")
    if unexpected:
        print(f"[run_eval] Unexpected keys ({len(unexpected)}): {list(unexpected)[:10]} ...")

    model.eval()
    return model


# -----------------------------
# 4. SmoothQuant / Naive W8A8
# -----------------------------
def maybe_apply_quantization(
    model: TinyRecursiveReasoningModel_ACTV1,
    calib_loader: DataLoader,
    device: str,
    use_smoothquant: bool,
    sq_alpha: float,
    sq_max_calib_batches: int,
):
    print(f"[run_eval] Quantization settings: use_smoothquant={use_smoothquant}, "
          f"sq_alpha={sq_alpha}, sq_max_calib_batches={sq_max_calib_batches}")

    if use_smoothquant:
        print("[quant_utils] === SmoothQuant W8A8 模型準備開始 ===")
        # 收集 calibration batch
        calib_batches = collect_calib_batches(calib_loader, sq_max_calib_batches)
        # 收集 activation 統計
        act_stats = calibrate_model(model, calib_batches, device)
        # 套用 SmoothQuant（重寫權重、計算 input_scale）
        input_scales = apply_smoothquant(model, act_stats, alpha=sq_alpha)
        # 轉成 W8A8Linear
        replace_with_quantized_layers(model, input_scales_map=input_scales)
    else:
        print("[quant_utils] === Naive W8A8 模型準備開始（無 SmoothQuant） ===")
        # 直接把所有 CastedLinear 換成 W8A8Linear（input_scale=1）
        replace_with_quantized_layers(model, input_scales_map=None)

    model.to(device)
    model.eval()


# -----------------------------
# 5. Evaluation：多步遞迴 ACT + Sudoku 指標
# -----------------------------
@torch.no_grad()
def evaluate_trm(
    model: TinyRecursiveReasoningModel_ACTV1,
    eval_loader: DataLoader,
    device: str,
    max_reasoning_steps: int,
    log_interval: int = 10,
) -> Dict[str, float]:
    total_tokens = 0
    correct_tokens = 0

    total_puzzles = 0
    solved_puzzles = 0

    total_steps_sum = 0.0
    total_halted_examples = 0

    step_idx = 0

    print("[run_eval] Start evaluation (multi-step ACT)...")

    for set_name, batch, global_effective_batch_size in eval_loader:
        # batch: dict(inputs, labels, puzzle_identifiers)
        batch_size = batch["inputs"].shape[0]

        # 移動到 device
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        # 初始 carry
        carry = model.initial_carry(batch_dev)

        final_logits = None

        # 多步遞迴推理：最多 max_reasoning_steps 步
        for t in range(max_reasoning_steps):
            carry, outputs = model(carry, batch_dev)
            final_logits = outputs["logits"]  # [B, S, V]

            if bool(carry.halted.all()):
                break

        # 統計 steps / halt 情況
        total_steps_sum += carry.steps.float().sum().item()
        total_halted_examples += carry.halted.sum().item()

        # 計算 token / puzzle accuracy
        logits = final_logits
        preds = logits.argmax(dim=-1)  # [B, S]
        labels = batch_dev["labels"]   # [B, S]

        mask = labels != IGNORE_LABEL_ID

        correct = (preds == labels) & mask

        correct_tokens += correct.sum().item()
        total_tokens += mask.sum().item()

        # per-puzzle （exact match）
        # 只要有一個 token 錯就算沒解對
        correct_per_puzzle = correct.view(batch_size, -1).all(dim=1)
        solved_puzzles += correct_per_puzzle.sum().item()
        total_puzzles += batch_size

        if step_idx % log_interval == 0:
            token_acc = correct_tokens / max(total_tokens, 1)
            puzzle_acc = solved_puzzles / max(total_puzzles, 1)
            print(f"[run_eval] step={step_idx}, token_acc={token_acc:.4f}, puzzle_acc={puzzle_acc:.4f}")

        step_idx += 1

    token_acc = correct_tokens / max(total_tokens, 1)
    puzzle_acc = solved_puzzles / max(total_puzzles, 1)
    avg_steps = total_steps_sum / max(total_puzzles, 1)
    halt_rate = total_halted_examples / max(total_puzzles * 1.0, 1.0)

    print(f"[run_eval] Final token-level accuracy:  {token_acc:.4f}")
    print(f"[run_eval] Final puzzle-level accuracy: {puzzle_acc:.4f}")
    print(f"[run_eval] Average reasoning steps per puzzle: {avg_steps:.2f}")
    print(f"[run_eval] Halt rate (final): {halt_rate:.4f}")

    return dict(
        token_accuracy=token_acc,
        exact_accuracy=puzzle_acc,
        avg_steps=avg_steps,
        halt_rate=halt_rate,
    )


# -----------------------------
# 6. main()
# -----------------------------
def main():
    cli_args = parse_cli_args()
    cfg = EvalCLIConfig(**cli_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_eval] Using device: {device}")

    # 讀 all_config.yaml
    all_cfg = load_all_config(cfg.load_checkpoint)
    seed = all_cfg.get("seed", 0)

    # Dataset & Dataloaders
    dataset, eval_loader, calib_loader = build_dataloaders(
        data_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        seed=seed,
    )

    # 建立 TRM Config
    model_config = build_trm_config(
        all_cfg=all_cfg,
        dataset=dataset,
        cli_args=cli_args,
        global_batch_size=cfg.global_batch_size,
    )

    # 載入模型
    model = load_trm_from_checkpoint(
        model_config=model_config,
        load_checkpoint=cfg.load_checkpoint,
        device=device,
    )

    # 量化（Naive W8A8 / SmoothQuant W8A8）
    maybe_apply_quantization(
        model=model,
        calib_loader=calib_loader,
        device=device,
        use_smoothquant=cfg.use_smoothquant,
        sq_alpha=cfg.sq_alpha,
        sq_max_calib_batches=cfg.sq_max_calib_batches,
    )

    # Evaluation（多步 ACT）
    max_steps = model_config.halt_max_steps
    results = evaluate_trm(
        model=model,
        eval_loader=eval_loader,
        device=device,
        max_reasoning_steps=max_steps,
    )

    # 存結果
    if cfg.checkpoint_path:
        os.makedirs(cfg.checkpoint_path, exist_ok=True)
        out_path = os.path.join(cfg.checkpoint_path, "eval_results.txt")
        with open(out_path, "w") as f:
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"[run_eval] Saved results to {out_path}")


if __name__ == "__main__":
    main()

