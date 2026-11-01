#!/bin/bash
# Local test eval of upstream SpaMo PHOENIX→PJM-MS ckpt (poster row "SpaMo PHOENIX MS = 5.46").
# Verifies the upstream test BLEU and adds BLEURT-20 (which the upstream paper didn't report).
# Waits for any running queue_bleurt to finish so GPU is free.
# Run: nohup bash scripts/queue_eval_upstream_phoenix_ms.sh > queue_phoenix_ms_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."

LOG=queue_phoenix_ms_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== Wait for any running eval queues: $(date) ==="
while pgrep -f "queue_bleurt_unisign|queue_eval_v28|fine_tuning\.py" > /dev/null; do
    sleep 30
done
echo "GPU free at $(date)."

source .venv/bin/activate

echo "=== Local eval: upstream SpaMo PHOENIX→PJM-MS ==="
python main.py \
    -c configs/spamo-pjm-phoenix-ms-test.yaml \
    --ckpt checkpoints/pheonix-spamo-pretrained.ckpt \
    -t False \
    --test True \
    -n test_upstream_phoenix_ms \
    --tags test pjm-ms dual phoenix-upstream || echo "(eval failed)"

echo "=== Done: $(date) ==="
