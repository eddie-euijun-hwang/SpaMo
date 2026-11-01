#!/bin/bash
# Sequential queue, waits for v30 to finish first:
#   0. Wait for finetune_v30 main.py process to exit
#   1. Uni-Sign MS scratch (no transfer init) — ~12h
#   2. v31 Mod-SpaMo quad-xattn MS PJM-only — ~12h
# Run with: nohup bash scripts/queue_unisign_scratch_then_v31.sh > queue_unisign_v31_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e
cd "$(dirname "$0")/.."

LOG=queue_unisign_v31_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== [0/2] Waiting for v30 (finetune_v30) to finish ==="
while pgrep -f "main.py.*finetune_v30" > /dev/null; do
    sleep 60
done
echo "v30 done at $(date)."

echo "=== [1/2] Uni-Sign MS scratch (no transfer init) ==="
( cd /home/croco/Uni-Sign && bash script/train_pjm_phase3_ms_scratch.sh ) || echo "(uni-sign scratch finished with non-zero, continuing)"

echo "=== [2/2] launching v31 (Mod-SpaMo quad-xattn MS PJM-only) ==="
source .venv/bin/activate
python main.py \
    -c configs/mod-spamo-pjm-scratch-ms.yaml \
    -e bleu \
    -n finetune_v31_quad_xattn_ms_scratch \
    --tags finetune pjm-ms quad scratch

echo "=== Queue done. ==="
