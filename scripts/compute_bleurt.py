"""Post-hoc BLEURT-20 over a `test_outputs.txt` dump (Reference:/Generated: blocks)."""

import argparse
import json
import re
from pathlib import Path

import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification
from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer

MODEL = "lucadiliello/BLEURT-20"


def parse_dump(path: Path) -> tuple[list[str], list[str]]:
    refs, gens = [], []
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^Reference:\s*(.*)$", line):
            cur = m.group(1).strip()
        elif m := re.match(r"^Generated:\s*(.*)$", line):
            assert cur is not None, f"Generated before Reference in {path}"
            refs.append(cur)
            gens.append(m.group(1).strip())
            cur = None
    return refs, gens


@torch.inference_mode()
def score(
    refs: list[str], gens: list[str], device: str, batch_size: int
) -> list[float]:
    tokenizer = BleurtSPTokenizer.from_pretrained(MODEL)
    model = BleurtForSequenceClassification.from_pretrained(
        MODEL, config=BleurtConfig.from_pretrained(MODEL)
    )
    model.eval().to(device)
    out = []
    for i in range(0, len(refs), batch_size):
        enc = tokenizer(
            refs[i : i + batch_size],
            gens[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    refs, gens = parse_dump(args.dump)
    scores = score(refs, gens, args.device, args.batch_size)
    mean = sum(scores) / len(scores)
    print(f"BLEURT-20 mean: {mean:.4f}  (n={len(scores)})")

    out_path = args.dump.parent / "bleurt_BLEURT-20.json"
    out_path.write_text(
        json.dumps({"n": len(scores), "mean": mean, "per_sample": scores})
    )


if __name__ == "__main__":
    main()
