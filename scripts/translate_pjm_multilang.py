"""
Translate PJM English texts (`features/texts_eng.h5`) to French + Spanish using NLLB-200.

Powód użycia angielskiego źródła zamiast polskiego (texts.h5): 86% PJM tekstów PL nie ma
diakrytyków — NLLB słabo radzi sobie z 'sloju'/'zaba' jako 'warstwa'/'zabawa' itd.
texts_eng.h5 to już czyste źródło, EN→FR/ES daje paper-grade quality.

Input:
    features/texts_eng.h5     — angielski MT (czysty)
    features/texts.h5         — polski oryginał (przepuszczany verbatim do `pl` field)
Output:
    features/texts_multilang_pjm.h5
        każdy fileid jako h5 group z polami /pl, /en, /fr, /es (UTF-8 strings)

Wybór modelu: facebook/nllb-200-distilled-600M
- offline, free, ~2.4GB
- eng_Latn → fra_Latn, eng_Latn → spa_Latn

Idempotentny: pomija fileid już obecne w wyjściowym H5.

Usage:
    python scripts/translate_pjm_multilang.py \\
        --src features/texts.h5 \\
        --en features/texts_eng.h5 \\
        --output features/texts_multilang_pjm.h5 \\
        --batch-size 32 \\
        --device cpu          # GPU zajęte przez trening
"""

import argparse
import logging
import time

import h5py
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def batch_translate(texts, tokenizer, model, src_lang, tgt_lang, device, max_len=128):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    ).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    out = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_length=max_len,
        num_beams=1,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="features/texts.h5")
    p.add_argument("--en", default="features/texts_eng.h5")
    p.add_argument("--output", default="features/texts_multilang_pjm.h5")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--cache-dir", default="/home/croco/.cache/huggingface/hub")
    p.add_argument(
        "--device", default="cpu", help="cpu (default — GPU zajęte) lub cuda:0"
    )
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    logger.info(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir)
        .to(args.device)
        .eval()
    )
    if args.device.startswith("cuda"):
        model = model.half()
    logger.info("Model loaded.")

    pl_h5 = h5py.File(args.src, "r")
    en_h5 = h5py.File(args.en, "r")
    out_h5 = h5py.File(args.output, "a")

    # Source for translation = English (texts_eng.h5). Iterate over EN keys; PL fetched if present.
    all_keys = sorted(en_h5.keys())
    todo = [k for k in all_keys if k not in out_h5]
    logger.info(
        f"Total EN keys: {len(all_keys):,} | Already done: {len(all_keys) - len(todo):,} | To do: {len(todo):,}"
    )

    if not todo:
        logger.info("Nothing to do.")
        return

    pbar = tqdm(total=len(todo), mininterval=10)
    start = time.time()
    flush_every = 500

    for i in range(0, len(todo), args.batch_size):
        batch_keys = todo[i : i + args.batch_size]
        en_texts = [en_h5[k][()].decode() for k in batch_keys]
        pl_texts = [pl_h5[k][()].decode() if k in pl_h5 else "" for k in batch_keys]

        try:
            fr_texts = batch_translate(
                en_texts, tokenizer, model, "eng_Latn", "fra_Latn", args.device
            )
            es_texts = batch_translate(
                en_texts, tokenizer, model, "eng_Latn", "spa_Latn", args.device
            )
        except Exception as e:
            logger.warning(f"Batch fail at {batch_keys[0]}: {e}")
            continue

        for k, pl, en, fr, es in zip(
            batch_keys, pl_texts, en_texts, fr_texts, es_texts
        ):
            grp = out_h5.create_group(k)
            grp.create_dataset("pl", data=pl)
            grp.create_dataset("en", data=en)
            grp.create_dataset("fr", data=fr)
            grp.create_dataset("es", data=es)

        pbar.update(len(batch_keys))
        if (i // args.batch_size) % (flush_every // args.batch_size + 1) == 0:
            out_h5.flush()
            elapsed = (time.time() - start) / 60
            done = i + len(batch_keys)
            logger.info(
                f"{done:,}/{len(todo):,} | {elapsed:.1f}min | rate {done / elapsed:.1f}/min"
            )

    pbar.close()
    out_h5.flush()
    out_h5.close()
    pl_h5.close()
    en_h5.close()

    elapsed = (time.time() - start) / 60
    logger.info(f"Done. Total time: {elapsed:.1f}min")


if __name__ == "__main__":
    main()
