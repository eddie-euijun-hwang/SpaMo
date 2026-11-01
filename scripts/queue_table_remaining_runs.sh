#!/bin/bash
# Train + eval the remaining runs needed for the poster table:
#   1. Uni-Sign PJM SI from scratch  → eval + post-hoc BLEURT-20
#   2. v32 SpaMo dual SI scratch     → test eval (inline BLEURT)
#   3. v33 Mod-SpaMo quad SI scratch → test eval (inline BLEURT)
# Waits for any running queues so GPU is free.
# Run: nohup bash scripts/queue_table_remaining_runs.sh > queue_table_remaining_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."

LOG=queue_table_remaining_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

UNISIGN_DIR=/home/croco/Uni-Sign

echo "=== Wait for any running queues: $(date) ==="
# Wait until GPU is idle (memory < 1 GiB), exclude this script's own children from any process check.
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1024 ]; do
    sleep 30
done
echo "GPU free at $(date)."

###############################################################################
# 1. Uni-Sign PJM SI from scratch
###############################################################################
US_OUT_TRAIN="$UNISIGN_DIR/out/pjm_phase3_full_si_scratch"
US_OUT_EVAL="$UNISIGN_DIR/out/eval_pjm_phase3_full_si_scratch"

if [ -f "$US_OUT_TRAIN/best_checkpoint.pth" ]; then
    echo "SKIP Uni-Sign SI scratch training — best_checkpoint.pth already exists"
else
    echo "=== [1a] Uni-Sign SI scratch train: $(date) ==="
    ( cd "$UNISIGN_DIR" && bash script/train_pjm_phase3_si_scratch.sh ) || echo "(uni-sign si scratch training failed, continuing)"
fi

if [ -f "$US_OUT_TRAIN/best_checkpoint.pth" ] && [ ! -f "$US_OUT_EVAL/test_tmp_refs.txt" ]; then
    echo "=== [1b] Uni-Sign SI scratch eval: $(date) ==="
    mkdir -p "$US_OUT_EVAL"
    PORT=29501
    while ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT$"; do
        PORT=$((PORT + 1))
    done
    echo "    using master port: $PORT"
    ( cd "$UNISIGN_DIR" && \
      source .venv/bin/activate && \
      torchrun --nproc_per_node=1 --master_port=$PORT fine_tuning.py \
          --eval \
          --finetune "$US_OUT_TRAIN/best_checkpoint.pth" \
          --dataset PJM \
          --task SLT \
          --pjm_split si \
          --bertscore \
          --num_examples 5 \
          --batch-size 8 \
          --num_workers 4 \
          --output_dir "$US_OUT_EVAL" \
          --wandb \
          --wandb_project uni-sign-eval \
          2>&1 | tee "$US_OUT_EVAL/log.txt" ) || echo "(uni-sign si scratch eval failed, continuing)"
fi

if [ -f "$US_OUT_EVAL/test_tmp_refs.txt" ] && [ ! -f "$US_OUT_EVAL/bleurt_BLEURT-20.json" ]; then
    echo "=== [1c] BLEURT-20 post-hoc Uni-Sign SI scratch: $(date) ==="
    source .venv/bin/activate
    python scripts/bleurt_unisign.py "$US_OUT_EVAL" 2>&1 | tee -a "$US_OUT_EVAL/log.txt" || echo "(bleurt failed)"
    deactivate
fi

###############################################################################
# 2. SpaMo v32 dual SI scratch
###############################################################################
echo "=== [2a] v32 dual SI scratch train: $(date) ==="
source .venv/bin/activate
python main.py \
    -c configs/spamo-pjm-scratch-si.yaml \
    -e bleu \
    -n finetune_v32_dual_si_scratch \
    --tags finetune pjm-si dual scratch || echo "(v32 train failed, continuing)"

# Pick best ckpt by parsing logs/<latest>_finetune_v32*/checkpoints
V32_LOG=$(ls -td logs/*finetune_v32_dual_si_scratch* 2>/dev/null | head -1)
V32_CKPT=$(ls -t "$V32_LOG"/checkpoints/epoch=*.ckpt 2>/dev/null | grep -v last | head -1)
if [ -n "$V32_CKPT" ]; then
    echo "=== [2b] v32 test eval: $V32_CKPT ==="
    python main.py \
        -c configs/spamo-pjm-scratch-si.yaml \
        --ckpt "$V32_CKPT" \
        -t False \
        --test True \
        -n test_v32_dual_si_scratch \
        --tags test pjm-si dual scratch || echo "(v32 test failed, continuing)"
else
    echo "SKIP v32 test — no best ckpt found in $V32_LOG"
fi

###############################################################################
# 3. Mod-SpaMo v33 quad-xattn SI scratch
###############################################################################
echo "=== [3a] v33 quad-xattn SI scratch train: $(date) ==="
python main.py \
    -c configs/mod-spamo-pjm-scratch-si.yaml \
    -e bleu \
    -n finetune_v33_quad_xattn_si_scratch \
    --tags finetune pjm-si quad scratch || echo "(v33 train failed, continuing)"

V33_LOG=$(ls -td logs/*finetune_v33_quad_xattn_si_scratch* 2>/dev/null | head -1)
V33_CKPT=$(ls -t "$V33_LOG"/checkpoints/epoch=*.ckpt 2>/dev/null | grep -v last | head -1)
if [ -n "$V33_CKPT" ]; then
    echo "=== [3b] v33 test eval: $V33_CKPT ==="
    python main.py \
        -c configs/mod-spamo-pjm-scratch-si.yaml \
        --ckpt "$V33_CKPT" \
        -t False \
        --test True \
        -n test_v33_quad_si_scratch \
        --tags test pjm-si quad scratch || echo "(v33 test failed, continuing)"
else
    echo "SKIP v33 test — no best ckpt found in $V33_LOG"
fi

deactivate
echo "=== Queue done: $(date) ==="
