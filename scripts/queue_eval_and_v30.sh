#!/bin/bash
# Sequential queue on local GPU:
#   1. Test eval v27 (best ckpt BLEU 5.40)
#   2. Uni-Sign eval_best_checkpoints (6 phase3 runs)
#   3. Launch v30 (dual MS from scratch)
set -e
cd "$(dirname "$0")/.."

LOG=queue_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== [1/2] Uni-Sign eval ==="
( cd /home/croco/Uni-Sign && bash scripts/eval_best_checkpoints.sh ) || echo "(uni-sign eval finished with non-zero, continuing)"

echo "=== [2/2] launching v30 (dual MS scratch) ==="
source .venv/bin/activate
python main.py \
    -c configs/spamo-pjm-scratch-ms.yaml \
    -e bleu \
    -n finetune_v30_dual_ms_scratch \
    --tags finetune pjm-ms dual scratch

echo "=== Queue done. ==="
