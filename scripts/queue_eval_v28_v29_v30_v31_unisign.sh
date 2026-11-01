#!/bin/bash
# Sequential test eval queue:
#   1. SpaMo v28 (dual SI from upstream)
#   2. SpaMo v29 (quad-xattn SI from upstream)
#   3. SpaMo v30 (dual MS scratch)
#   4. SpaMo v31 (quad-xattn MS scratch)
#   5. Uni-Sign PJM MS scratch (no transfer init)
# SpaMo runs compute BLEURT-20 inline via utils/evaluate.py.
# Uni-Sign run gets post-hoc BLEURT-20 via scripts/bleurt_unisign.py.
# Run: nohup bash scripts/queue_eval_v28_v29_v30_v31_unisign.sh > queue_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."

LOG=queue_eval_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== Eval queue start: $(date) ==="

source .venv/bin/activate

run_spamo_test() {
    local cfg="$1"; local ckpt="$2"; local name="$3"; shift 3
    echo "=== SpaMo test: $name ==="
    echo "    cfg=$cfg"
    echo "    ckpt=$ckpt"
    python main.py \
        -c "$cfg" \
        --ckpt "$ckpt" \
        -t False \
        --test True \
        -n "$name" \
        --tags "$@"
    echo "=== Done $name at $(date) ==="
}

# 1. v28 — dual SI from upstream
run_spamo_test \
    configs/finetune_v28_dual_si_from_upstream.yaml \
    remote_ckpts/v28/epoch=00014-step=0020850-bleu4=0.52.ckpt \
    test_v28_dual_si \
    test pjm-si dual upstream-init || echo "(v28 failed, continuing)"

# 2. v29 — quad-xattn SI from upstream
run_spamo_test \
    configs/finetune_v29_quad_xattn_si_from_upstream.yaml \
    remote_ckpts/v29/epoch=00011-step=0016680-bleu4=0.52.ckpt \
    test_v29_quad_si \
    test pjm-si quad upstream-init || echo "(v29 failed, continuing)"

# 3. v30 — dual MS scratch
run_spamo_test \
    configs/spamo-pjm-scratch-ms.yaml \
    logs/2026-05-04T10-59-07_finetune_v30_dual_ms_scratch/checkpoints/epoch=00028-step=0085579-bleu4=1.97.ckpt \
    test_v30_dual_ms_scratch \
    test pjm-ms dual scratch || echo "(v30 failed, continuing)"

# 4. v31 — quad-xattn MS scratch
run_spamo_test \
    configs/mod-spamo-pjm-scratch-ms.yaml \
    logs/2026-05-05T09-06-44_finetune_v31_quad_xattn_ms_scratch/checkpoints/epoch=00009-step=0029510-bleu4=1.84.ckpt \
    test_v31_quad_ms_scratch \
    test pjm-ms quad scratch || echo "(v31 failed, continuing)"

deactivate

# 5. Uni-Sign PJM MS scratch
echo "=== Uni-Sign test: pjm_phase3_full_ms_scratch ==="
UNISIGN_DIR=/home/croco/Uni-Sign
US_CKPT="$UNISIGN_DIR/out/pjm_phase3_full_ms_scratch/best_checkpoint.pth"
US_OUT="$UNISIGN_DIR/out/eval_pjm_phase3_full_ms_scratch"
if [ ! -f "$US_CKPT" ]; then
    echo "SKIP — checkpoint not found: $US_CKPT"
elif [ -f "$US_OUT/log.txt" ]; then
    echo "SKIP — already evaluated ($US_OUT/log.txt exists)"
else
    mkdir -p "$US_OUT"
    PORT=29501
    while ss -tln 2>/dev/null | awk '{print $4}' | grep -qE ":$PORT$"; do
        PORT=$((PORT + 1))
    done
    echo "    using master port: $PORT"
    ( cd "$UNISIGN_DIR" && \
      source .venv/bin/activate && \
      torchrun --nproc_per_node=1 --master_port=$PORT fine_tuning.py \
          --eval \
          --finetune "$US_CKPT" \
          --dataset PJM \
          --task SLT \
          --pjm_split ms \
          --bertscore \
          --num_examples 5 \
          --batch-size 8 \
          --num_workers 4 \
          --output_dir "$US_OUT" \
          --wandb \
          --wandb_project uni-sign-eval \
          2>&1 | tee "$US_OUT/log.txt" ) || echo "(uni-sign eval failed, continuing)"
    echo "=== Done Uni-Sign MS scratch at $(date) ==="
fi

# 6. Post-hoc BLEURT-20 for Uni-Sign MS scratch
if [ -f "$US_OUT/test_tmp_refs.txt" ] && [ ! -f "$US_OUT/bleurt_BLEURT-20.json" ]; then
    echo "=== BLEURT-20 (post-hoc) for Uni-Sign MS scratch ==="
    source .venv/bin/activate
    python scripts/bleurt_unisign.py "$US_OUT" 2>&1 | tee -a "$US_OUT/log.txt" || echo "(bleurt post-hoc failed)"
    deactivate
fi

echo ""
echo "=== Summary ==="
for d in logs/*test_v28_dual_si* logs/*test_v29_quad_si* logs/*test_v30_dual_ms_scratch* logs/*test_v31_quad_ms_scratch*; do
    [ -d "$d" ] || continue
    echo "  $d"
done
[ -f "$US_OUT/log.txt" ] && echo "  $US_OUT"
echo "=== Eval queue done: $(date) ==="
