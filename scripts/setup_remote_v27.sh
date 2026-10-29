#!/bin/bash
# Setup script for v27 quad run on a remote machine.
# Assumes: Python 3.10, CUDA 11.8+, ~100GB disk, 24GB+ VRAM.
#
# Usage:
#   git clone https://github.com/croco-corp/SpaMo-PJM.git
#   cd SpaMo-PJM
#   bash scripts/setup_remote_v27.sh

set -e

echo "=== [0/5] Auth tokens ==="
echo "Required: HF_TOKEN (croco-corp private repos) and WANDB_API_KEY."
echo "Set them beforehand or enter now:"
echo ""
if [ -z "$HF_TOKEN" ]; then
    echo -n "HuggingFace token: "
    read -s HF_TOKEN
    echo ""
    export HF_TOKEN
fi
if [ -z "$WANDB_API_KEY" ]; then
    echo -n "W&B API key: "
    read -s WANDB_API_KEY
    echo ""
    export WANDB_API_KEY
fi

echo "=== [1/5] uv + dependencies ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install "setuptools<70"
uv pip install -e .

echo "=== [2/5] W&B login ==="
wandb login --relogin "$WANDB_API_KEY"

echo "=== [3/5] HuggingFace model weights ==="
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, RobertaModel, RobertaTokenizer
print("Downloading flan-t5-xl (~11GB)...")
AutoTokenizer.from_pretrained("google/flan-t5-xl")
AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
print("Downloading roberta-large (~1.4GB)...")
RobertaTokenizer.from_pretrained("roberta-large")
RobertaModel.from_pretrained("roberta-large")
print("Done.")
EOF

echo "=== [4/5] PJM feature files + checkpoint ==="
mkdir -p features checkpoints data/splits
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

def dl(repo, filename, local, repo_type="dataset"):
    if os.path.exists(local):
        print(f"  skip {local} (exists)")
        return
    print(f"  downloading {filename}...")
    p = hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type,
                        local_dir_use_symlinks=False, local_dir="/tmp/hf_dl")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    shutil.move(p, local)
    print(f"  {local} OK")

FEAT = "croco-corp/pjm-data"
dl(FEAT, "vectors/vit_feat_pjm.h5",          "features/vit_feat_pjm.h5")
dl(FEAT, "vectors/mae_feat_pjm.h5",          "features/mae_feat_pjm.h5")
dl(FEAT, "vectors/hand_vit_feat_pjm.h5",     "features/hand_vit_feat_pjm.h5")
dl(FEAT, "vectors/mediapipe_feat_pjm.h5",    "features/mediapipe_feat_pjm.h5")
dl(FEAT, "vectors/texts_eng.h5",             "features/texts_eng.h5")
dl(FEAT, "vectors/texts_multilang_pjm.h5",   "features/texts_multilang_pjm.h5")
dl(FEAT, "splits/split_train_ms.csv",        "data/splits/split_train_ms.csv")
dl(FEAT, "splits/split_val_ms.csv",          "data/splits/split_val_ms.csv")
dl(FEAT, "splits/split_test_ms.csv",         "data/splits/split_test_ms.csv")
dl(FEAT, "splits/split_train.csv",           "data/splits/split_train.csv")
dl(FEAT, "splits/split_val.csv",             "data/splits/split_val.csv")
dl(FEAT, "splits/split_test.csv",            "data/splits/split_test.csv")

CKPT = "CrocoSign/pjm-models"
dl(CKPT, "checkpoints/spamo/spamo_pjm_ms_dual_bleu5.36.ckpt", "checkpoints/spamo_pjm_ms_dual_bleu5.36.ckpt",
   repo_type="model")
EOF

echo "=== [5/5] Fix config paths + verify ==="
for cfg in configs/finetune_v27_quad_xattn_from_upstream.yaml \
           configs/finetune_v26_dual_paper_faithful_from_upstream.yaml \
           configs/finetune_v28_dual_si_from_upstream.yaml \
           configs/finetune_v29_quad_xattn_si_from_upstream.yaml; do
    sed -i 's|/home/croco/.cache/huggingface/hub|'"$HOME"'/.cache/huggingface/hub|g' "$cfg"
    sed -i 's|/home/croco/CrocoSign/data/split_train_ms\.csv|data/splits/split_train_ms.csv|g' "$cfg"
    sed -i 's|/home/croco/CrocoSign/data/split_val_ms\.csv|data/splits/split_val_ms.csv|g' "$cfg"
    sed -i 's|/home/croco/CrocoSign/data/split_test_ms\.csv|data/splits/split_test_ms.csv|g' "$cfg"
    sed -i 's|/home/croco/CrocoSign/data/split_train\.csv|data/splits/split_train.csv|g' "$cfg"
    sed -i 's|/home/croco/CrocoSign/data/split_val\.csv|data/splits/split_val.csv|g' "$cfg"
    sed -i 's|/home/croco/CrocoSign/data/split_test\.csv|data/splits/split_test.csv|g' "$cfg"
done

python - <<'EOF'
import os
required = [
    "features/vit_feat_pjm.h5",
    "features/mae_feat_pjm.h5",
    "features/hand_vit_feat_pjm.h5",
    "features/mediapipe_feat_pjm.h5",
    "features/texts_eng.h5",
    "features/texts_multilang_pjm.h5",
    "data/splits/split_train_ms.csv",
    "data/splits/split_val_ms.csv",
    "data/splits/split_test_ms.csv",
    "data/splits/split_train.csv",
    "data/splits/split_val.csv",
    "data/splits/split_test.csv",
    "checkpoints/spamo_pjm_ms_dual_bleu5.36.ckpt",
]
ok = True
for f in required:
    if os.path.exists(f):
        size = os.path.getsize(f) // (1024**2)
        print(f"  OK  {f} ({size} MB)")
    else:
        print(f"  MISSING  {f}")
        ok = False
if ok:
    print("\nAll files present. Ready to run v27.")
else:
    print("\nSome files missing — check errors above.")
    exit(1)
EOF

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo ""
echo "=== Setup complete. Detected ${NGPU} GPU(s). ==="

source .venv/bin/activate

if [ "$NGPU" -ge 2 ]; then
    echo "Running v28 (GPU 0) and v29 (GPU 1) in parallel."
    CUDA_VISIBLE_DEVICES=0 nohup python main.py \
        -c configs/finetune_v28_dual_si_from_upstream.yaml \
        --ckpt checkpoints/spamo_pjm_ms_dual_bleu5.36.ckpt \
        -e bleu -n finetune_v28_dual_si \
        --tags finetune pjm-si dual upstream-init \
        > nohup_v28.out 2>&1 &
    echo "v28 PID: $!"

    CUDA_VISIBLE_DEVICES=1 nohup python main.py \
        -c configs/finetune_v29_quad_xattn_si_from_upstream.yaml \
        --ckpt checkpoints/spamo_pjm_ms_dual_bleu5.36.ckpt \
        -e bleu -n finetune_v29_quad_xattn_si \
        --tags finetune pjm-si quad xattn upstream-init \
        > nohup_v29.out 2>&1 &
    echo "v29 PID: $!"
else
    echo "Single GPU — running v28 then v29 sequentially."
    python main.py \
        -c configs/finetune_v28_dual_si_from_upstream.yaml \
        --ckpt checkpoints/spamo_pjm_ms_dual_bleu5.36.ckpt \
        -e bleu -n finetune_v28_dual_si \
        --tags finetune pjm-si dual upstream-init \
        2>&1 | tee nohup_v28.out
    python main.py \
        -c configs/finetune_v29_quad_xattn_si_from_upstream.yaml \
        --ckpt checkpoints/spamo_pjm_ms_dual_bleu5.36.ckpt \
        -e bleu -n finetune_v29_quad_xattn_si \
        --tags finetune pjm-si quad xattn upstream-init \
        2>&1 | tee nohup_v29.out
fi
