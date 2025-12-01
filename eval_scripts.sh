#!/usr/bin/env bash

# Run evaluations for 5 pretrained models
# Each command runs on 8 GPUs using torchrun

# 1. ARC-1 concept evaluation
#torchrun --nproc-per-node=8 run_eval.py \
#  arch=trm \
#  data_paths="[data/arc1concept-aug-1000]" \
#  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
#  load_checkpoint=./checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_att_arc1concept_8/step_86510 \
#  checkpoint_path=./eval_results/arc1_att

# 2. ARC-2 concept evaluation
#torchrun --nproc-per-node=8 run_eval.py \
#  arch=trm \
#  data_paths="[data/arc2concept-aug-1000]" \
#  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
#  load_checkpoint=./checkpoints/Arc2concept-aug-1000-ACT-torch/pretrain_att_arc2concept_8/step_84623 \
#  checkpoint_path=./eval_results/arc2_att

# 3. Maze 30x30 evaluation (no evaluator)
#torchrun --nproc-per-node=8 run_eval.py \
#  arch=trm \
#  data_paths="[data/maze-30x30-hard-1k]" \
#  evaluators="[]" \
#  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
#  load_checkpoint=./checkpoints/Maze-30x30-hard-1k-ACT-torch/pretrain_att_maze30x30/step_9765 \
#  checkpoint_path=./eval_results/maze30x30_att

# 4. Sudoku (MLP-t version, no evaluator)
#torchrun --nproc-per-node=8 run_eval.py \
#  arch=trm \
#  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
#  evaluators="[]" \
#  arch.mlp_t=True arch.pos_encodings=none \
#  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
#  load_checkpoint=./checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_16275 \
#  checkpoint_path=./eval_results/sudoku_mlp_t

# 5. Sudoku (Attention version, no evaluator)
#torchrun --standalone --nproc-per-node=1 run_eval.py \
#  arch=trm \
#  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
#  evaluators="[]" \
#  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
#  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
#  checkpoint_path=./eval_results/sudoku_att_fp_baseline \
#  global_batch_size=128 \

# SmoothQuant
#torchrun --standalone --nproc-per-node=1 run_eval.py \
#  arch=trm \
#  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
#  evaluators="[]" \
#  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
#  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
#  checkpoint_path=./eval_results/sudoku_att_smoothquant_a05 \
#  global_batch_size=128 \
#  use_smoothquant=true \
#  sq_alpha=0.5 \
#  sq_max_calib_batches=128

# BF16 baseline
# torchrun --standalone --nproc-per-node=1 run_eval.py \
#   arch=trm \
#   data_paths="[data/sudoku-extreme-1k-aug-1000]" \
#   load_checkpoint=./checkpoints/sudoku_att/step_21700 \
#   checkpoint_path=./eval_results/sudoku_att_bf16_baseline \
#   global_batch_size=128 
  
# # Naive w8a8, non-smoothquant
torchrun --standalone --nproc-per-node=1 run_eval_v2.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
  checkpoint_path=./eval_results/sudoku_att_naive_w8a8 \
  global_batch_size=128 \
  use_smoothquant=false 

# # Smoothquant, alpha = 0.9
torchrun --standalone --nproc-per-node=1 run_eval_v2.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
  checkpoint_path=./eval_results/sudoku_att_smoothquant_a09 \
  global_batch_size=128 \
  use_smoothquant=true \
  sq_alpha=0.9 \
  sq_max_calib_batches=128
  
# # Smoothquant, alpha = 0.7
torchrun --standalone --nproc-per-node=1 run_eval_v2.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
  checkpoint_path=./eval_results/sudoku_att_smoothquant_a07 \
  global_batch_size=128 \
  use_smoothquant=true \
  sq_alpha=0.7 \
  sq_max_calib_batches=128

# # Smoothquant, alpha = 0.5
torchrun --standalone --nproc-per-node=1 run_eval_v2.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
  checkpoint_path=./eval_results/sudoku_att_smoothquant_a05 \
  global_batch_size=128 \
  use_smoothquant=true \
  sq_alpha=0.5 \
  sq_max_calib_batches=128

# # Smoothquant, alpha = 0.3
torchrun --standalone --nproc-per-node=1 run_eval_v2.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
  checkpoint_path=./eval_results/sudoku_att_smoothquant_a03 \
  global_batch_size=128 \
  use_smoothquant=true \
  sq_alpha=0.3 \
  sq_max_calib_batches=128

# Smoothquant, alpha = 0.1
torchrun --standalone --nproc-per-node=1 run_eval_v2.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  load_checkpoint=./checkpoints/sudoku_att/step_21700 \
  checkpoint_path=./eval_results/sudoku_att_smoothquant_a01 \
  global_batch_size=128 \
  use_smoothquant=true \
  sq_alpha=0.1 \
  sq_max_calib_batches=128