"""Post-hoc BLEURT-20 for Uni-Sign eval dumps (test_tmp_refs.txt / test_tmp_pres.txt)."""

import argparse
import json
import re
from pathlib import Path

import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification
from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer

MODEL = "lucadiliello/BLEURT-20"
REF_RE = re.compile(r"^sample:\s*(\S+),\s*ground-truth:\s*(.*)$")
PRE_RE = re.compile(r"^sample:\s*(\S+),\s*prediction:\s*(.*)$")


def parse(path: Path, regex: re.Pattern) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := regex.match(line):
            out[m.group(1)] = m.group(2).strip()
    return out


@torch.inference_mode()
def score(
    refs: list[str], gens: list[str], device: str, batch_size: int
) -> list[float]:
    tok = BleurtSPTokenizer.from_pretrained(MODEL)
    model = (
        BleurtForSequenceClassification.from_pretrained(
            MODEL, config=BleurtConfig.from_pretrained(MODEL)
        )
        .eval()
        .to(device)
    )
    out: list[float] = []
    for i in range(0, len(refs), batch_size):
        enc = tok(
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
    ap.add_argument(
        "eval_dir",
        type=Path,
        help="Uni-Sign eval output dir (contains test_tmp_refs.txt / test_tmp_pres.txt)",
    )
    ap.add_argument("--phase", default="test", help="phase prefix (test/dev)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    refs_map = parse(args.eval_dir / f"{args.phase}_tmp_refs.txt", REF_RE)
    pres_map = parse(args.eval_dir / f"{args.phase}_tmp_pres.txt", PRE_RE)
    common = sorted(set(refs_map) & set(pres_map))
    if not common:
        raise SystemExit(f"no overlapping samples in {args.eval_dir}")
    refs = [refs_map[k] for k in common]
    gens = [pres_map[k] for k in common]

    scores = score(refs, gens, args.device, args.batch_size)
    mean = sum(scores) / len(scores)
    print(f"BLEURT-20 mean: {mean:.4f}  (n={len(scores)})")

    out_path = args.eval_dir / "bleurt_BLEURT-20.json"
    out_path.write_text(
        json.dumps(
            {"n": len(scores), "mean": mean, "per_sample": dict(zip(common, scores))}
        )
    )


if __name__ == "__main__":
    main()
