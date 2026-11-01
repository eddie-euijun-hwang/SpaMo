"""
Hand-crop ViT feature extraction for PJM corpus.

For each frame: detects hands via MediaPipe, crops each hand region (with padding),
runs CLIP ViT-L/14 on the crop. Aggregates left+right hand embeddings by mean pooling.
Output shape per video: [T, 2048]  (same dim as main vit_feat_pjm.h5 for easy fusion)

If no hand detected in a frame, uses zero vector.

Architecture: N CPU workers do MediaPipe detection → queue → GPU writer encodes ViT.

Usage:
    uv run python scripts/hand_crop_vit_extract.py \
        --tars-dir /home/croco/CrocoSign/data/pjm_segments \
        --crop-params-path crop_params/crop_params.lmdb \
        --output features/hand_vit_feat_pjm.h5 \
        --num-workers 8
"""

import argparse
import io
import logging
import os
import tarfile
import time
from multiprocessing import Process, Queue

import h5py
import lmdb
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import msgpack
import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import pad as resize_and_pad
from av import open as av_open
from transformers import AutoImageProcessor, CLIPVisionModel
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HAND_CROP_PAD = 0.25
FEAT_DIM = 2048
QUEUE_MAXSIZE = 128  # max buffered videos between CPU and GPU


def get_video_frames(video_bytes: bytes) -> list:
    container = av_open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    return [frame.to_image() for frame in container.decode(stream)]


def crop_frame(
    image: Image.Image, crop_params: dict, output_size=(224, 224)
) -> Image.Image:
    cropped = image.crop(
        (
            crop_params["x_start"],
            crop_params["y_start"],
            crop_params["x_end"],
            crop_params["y_end"],
        )
    )
    return resize_and_pad(cropped, output_size)


def hand_bbox(landmarks, img_w: int, img_h: int, pad: float = HAND_CROP_PAD):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    return (
        int(max(0, x1 - pw)),
        int(max(0, y1 - ph)),
        int(min(img_w, x2 + pw)),
        int(min(img_h, y2 + ph)),
    )


def iter_tar_names(tars_dir: str, tar_names: list):
    for tar_name in tar_names:
        tar_path = os.path.join(tars_dir, tar_name)
        with tarfile.open(tar_path) as tf:
            members = {m.name: m for m in tf.getmembers()}
            mp4_keys = [n[:-4] for n in members if n.endswith(".mp4")]
            for key in mp4_keys:
                f = tf.extractfile(members[key + ".mp4"])
                if f is not None:
                    yield key, f.read()


def cpu_worker(
    worker_id: int,
    tar_names: list,
    tars_dir: str,
    crop_params_path: str,
    out_queue: Queue,
    already_done: set,
    hand_model_path: str,
):
    """CPU worker: detects hand crops and puts numpy arrays into the queue."""
    log = logging.getLogger(f"cpu-{worker_id}")
    try:
        _cpu_worker_inner(
            worker_id,
            tar_names,
            tars_dir,
            crop_params_path,
            out_queue,
            already_done,
            log,
            hand_model_path,
        )
    except Exception as e:
        log.error(f"Worker {worker_id} crashed: {e}")
    finally:
        out_queue.put(("done", worker_id))


def _cpu_worker_inner(
    worker_id,
    tar_names,
    tars_dir,
    crop_params_path,
    out_queue,
    already_done,
    log,
    hand_model_path,
):
    hands_detector = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2,
            running_mode=RunningMode.IMAGE,
        )
    )
    lmdb_env = lmdb.open(crop_params_path, readonly=True, lock=False)

    for key, mp4_bytes in iter_tar_names(tars_dir, tar_names):
        if key in already_done:
            out_queue.put(("skip", key))
            continue
        try:
            frames = get_video_frames(mp4_bytes)
            with lmdb_env.begin(write=False) as txn:
                raw = txn.get(key.encode())
            if raw is None:
                raise ValueError(f"No crop params for {key}")
            crop_params = msgpack.unpackb(raw)

            crops_list = []  # flat list of np.ndarray [H, W, 3]
            frame_sizes = []  # number of hand crops per frame
            for frame in frames:
                cropped = crop_frame(frame, crop_params)
                rgb = np.array(cropped.convert("RGB"))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hands_detector.detect(mp_img)
                h, w = rgb.shape[:2]
                n = 0
                if result.hand_landmarks:
                    for hand_lm in result.hand_landmarks:
                        x1, y1, x2, y2 = hand_bbox(hand_lm, w, h)
                        if x2 > x1 and y2 > y1:
                            crops_list.append(rgb[y1:y2, x1:x2].copy())
                            n += 1
                frame_sizes.append(n)

            out_queue.put(("ok", key, crops_list, frame_sizes))

        except Exception as e:
            out_queue.put(("error", key, str(e)))

    lmdb_env.close()
    hands_detector.close()
    log.info("Worker finished.")


@torch.no_grad()
def encode_crops(crops: list, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=crops, return_tensors="pt").to(device)
    hidden = model(**inputs, output_hidden_states=True).hidden_states[
        -1
    ]  # [N, seq, 1024]
    cls = hidden[:, 0]  # [N, 1024]
    patch_mean = hidden[:, 1:].mean(1)  # [N, 1024]
    return torch.cat([cls, patch_mean], dim=-1).float().cpu().numpy()  # [N, 2048]


def gpu_writer(out_queue: Queue, n_workers: int, args, processed: set):
    """Main process: encodes hand crops with ViT, writes results to H5."""
    logger.info("Loading CLIP ViT-L/14...")
    processor = AutoImageProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir
    )
    vit = (
        CLIPVisionModel.from_pretrained(
            args.model_name,
            output_hidden_states=True,
            cache_dir=args.cache_dir,
        )
        .to(args.device)
        .eval()
    )
    logger.info("Model loaded. GPU writer ready.")

    zero_feat = np.zeros(FEAT_DIM, dtype=np.float32)
    done_workers = 0
    newly = errors = skipped = 0
    start = time.time()

    pbar = tqdm.tqdm(desc="hand-vit", mininterval=30)

    with h5py.File(args.output, "a") as hf:
        hf.attrs.setdefault("feat_dim", FEAT_DIM)
        hf.attrs.setdefault(
            "description",
            "CLIP ViT-L/14 on hand crops, mean-pooled L+R, zero if no detection",
        )

        while done_workers < n_workers:
            item = out_queue.get()
            tag = item[0]

            if tag == "done":
                done_workers += 1
                continue

            if tag == "skip":
                skipped += 1
                pbar.update(1)
                pbar.set_postfix(new=newly, skip=skipped, err=errors)
                continue

            if tag == "error":
                _, key, msg = item
                errors += 1
                logger.warning(f"Error {key}: {msg}")
                pbar.update(1)
                pbar.set_postfix(new=newly, skip=skipped, err=errors)
                continue

            _, key, crops_list, frame_sizes = item

            if crops_list:
                all_pil = [Image.fromarray(c) for c in crops_list]
                all_feats = []
                for i in range(0, len(all_pil), args.batch_size):
                    batch = all_pil[i : i + args.batch_size]
                    all_feats.append(encode_crops(batch, processor, vit, args.device))
                all_feats = np.concatenate(all_feats, axis=0)
            else:
                all_feats = np.empty((0, FEAT_DIM), dtype=np.float32)

            video_feats = []
            idx = 0
            for n in frame_sizes:
                if n > 0:
                    video_feats.append(all_feats[idx : idx + n].mean(axis=0))
                    idx += n
                else:
                    video_feats.append(zero_feat)

            arr = np.stack(video_feats, axis=0)
            hf.create_dataset(key, data=arr, dtype="float32")
            newly += 1
            pbar.update(1)
            pbar.set_postfix(new=newly, skip=skipped, err=errors)

            if newly % args.flush_every == 0:
                hf.flush()
                elapsed = time.time() - start
                logger.info(
                    f"{newly + len(processed)}/{newly + skipped + len(processed)} | {elapsed / 60:.1f}min"
                )

        hf.flush()

    pbar.close()
    elapsed = time.time() - start
    logger.info(
        f"Done. Newly: {newly:,} | Errors: {errors} | Time: {elapsed / 60:.1f} min"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tars-dir", default="/home/croco/CrocoSign/data/pjm_segments")
    parser.add_argument("--crop-params-path", default="crop_params/crop_params.lmdb")
    parser.add_argument("--output", default="features/hand_vit_feat_pjm.h5")
    parser.add_argument("--cache-dir", default="/home/croco/.cache/huggingface/hub")
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=500)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() - 2),
        help="CPU workers for MediaPipe detection (default: nproc-2)",
    )
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count videos and exit without processing",
    )
    args = parser.parse_args()

    all_tars = sorted(f for f in os.listdir(args.tars_dir) if f.endswith(".tar"))

    total = sum(
        sum(
            1
            for m in tarfile.open(os.path.join(args.tars_dir, f)).getmembers()
            if m.name.endswith(".mp4")
        )
        for f in all_tars
    )
    logger.info(f"Total videos: {total:,}")

    processed = set()
    if os.path.exists(args.output):
        with h5py.File(args.output, "r") as f:
            processed = set(f.keys())
        logger.info(f"Resuming — {len(processed):,} already done")

    remaining = total - len(processed)
    logger.info(f"To process: {remaining:,} | CPU workers: {args.num_workers}")

    if args.dry_run:
        logger.info("Dry run — exiting without processing.")
        return

    # Split tars across CPU workers (round-robin)
    chunks = [all_tars[i :: args.num_workers] for i in range(args.num_workers)]
    out_queue = Queue(maxsize=QUEUE_MAXSIZE)

    # Spawn CPU workers first — before any CUDA init in this process
    cpu_procs = []
    for i, chunk in enumerate(chunks):
        p = Process(
            target=cpu_worker,
            args=(
                i,
                chunk,
                args.tars_dir,
                args.crop_params_path,
                out_queue,
                processed,
                args.hand_model,
            ),
            daemon=True,
        )
        p.start()
        cpu_procs.append(p)

    # GPU writer runs in main process (safe: CUDA init after fork)
    gpu_writer(out_queue, args.num_workers, args, processed)

    for p in cpu_procs:
        p.join()


if __name__ == "__main__":
    main()
