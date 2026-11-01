import logging
from typing import Callable
import sys
import argparse
import os.path as osp

import h5py
import numpy as np
import torch
import tqdm
import lmdb
import datasets
from transformers import VideoMAEModel, VideoMAEImageProcessor

sys.path.append("./")

from utils.pjm.preprocessing import ImageConverter, get_video_frames
from utils.helpers import sliding_window_for_list
from utils.log import get_error_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

error_logger = get_error_logger("mae_features_extract_errors.log")

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True  # type: ignore


class VideoMAEFeatureReader(object):
    def __init__(
        self,
        model_name="MCG-NJU/videomae-large",
        cache_dir=None,
        device="cuda:0",
        overlap_size=0,
        nth_layer=-1,
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        logger.info(f"Loading VideoMAE model: {model_name}")
        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()  # type: ignore
        logger.info(f"Model loaded successfully on {self.device}")

    @torch.no_grad()
    def get_feats(self, chunk_batch):
        inputs = self.image_processor(images=chunk_batch, return_tensors="pt").to(
            self.device
        )  # type: ignore

        outputs = self.model(**inputs, output_hidden_states=True).hidden_states

        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]

        return outputs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        help="path to huggingface dataset",
        type=str,
        default="croco-corp/pjm-segments",
    )
    parser.add_argument(
        "--crop-params-path",
        help="path to crop params lmdb file",
        type=str,
        default="crop_params/crop_params.lmdb",
    )
    parser.add_argument("--device", help="device to use", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overlap-size", type=int, default=8)
    parser.add_argument(
        "--nth-layer", help="which layer to extract features from", type=int, default=-1
    )
    parser.add_argument(
        "--cache-dir", help="cache directory for model", type=str, default=None
    )
    parser.add_argument(
        "--model-name",
        help="ViT model name",
        type=str,
        default="MCG-NJU/videomae-large",
    )
    parser.add_argument(
        "--save-dir", help="where to save the output", type=str, required=True
    )
    parser.add_argument(
        "--split",
        help="dataset split to process",
        type=str,
        default="train",
        choices=["train"],  # ["train", "val", "test"]
    )
    return parser


def get_iterator_for_pjm(
    args, split: str = "train", skip_ids: set = None
) -> tuple[Callable, int]:
    batch_size = args.batch_size
    logger.info(f"Loading dataset: {args.dataset_path}, split: {split}")
    dataset = datasets.load_dataset(
        args.dataset_path, split=split, cache_dir=args.cache_dir
    )
    num = len(dataset)

    logger.info(f"Dataset loaded: {num} videos")

    if skip_ids:
        logger.info(f"Will skip {len(skip_ids)} already processed videos")

    logger.info(f"Opening LMDB crop params: {args.crop_params_path}")
    lmdb_env = lmdb.open(args.crop_params_path, readonly=True, lock=False)
    converter = ImageConverter(lmdb_env, (224, 224))

    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
    )

    def iterate():
        skipped_count = 0
        processed_count = 0
        error_count = 0
        pbar = tqdm.tqdm(total=num, desc="Processing PJM")
        REQUIRED_NUM_FRAMES_FOR_MAE = 16
        for rec in dataset:
            try:
                video_id = rec["__key__"]

                if skip_ids and video_id in skip_ids:
                    skipped_count += 1
                    continue

                frames = get_video_frames(rec["mp4"])
                if not frames:
                    raise Exception("No frames found")

                processed_frames = converter.process_frames(frames, video_id)

                if len(processed_frames) < REQUIRED_NUM_FRAMES_FOR_MAE:
                    processed_frames.extend(
                        [processed_frames[-1]]
                        * (REQUIRED_NUM_FRAMES_FOR_MAE - len(processed_frames))
                    )

                frame_chunks = sliding_window_for_list(
                    processed_frames,
                    window_size=REQUIRED_NUM_FRAMES_FOR_MAE,
                    overlap_size=args.overlap_size,
                )

                chunk_feats = []
                for j in range(0, len(frame_chunks), batch_size):
                    chunk_batch = frame_chunks[j : j + batch_size]
                    feats = reader.get_feats(chunk_batch).cpu().numpy()
                    chunk_feats.append(feats)

                processed_count += 1
                yield (
                    np.concatenate(chunk_feats, axis=0),
                    video_id,
                    None,
                )  # For now None, maybe in the future start time
            except Exception:
                error_count += 1
                failed_video_id = rec["__key__"]
                error_logger.exception(f"extract features error for {failed_video_id}")
            finally:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "processed": processed_count,
                        "skipped": skipped_count,
                        "errors": error_count,
                    }
                )

        lmdb_env.close()
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} videos total")
        logger.info(f"Successfully processed {processed_count} videos")
        logger.info(f"failed videos: {error_count}")

    return iterate, num


def save_hdf5(save_in_every: int = 500):
    import time

    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()
    split = args.split

    logger.info("=" * 60)
    logger.info("MAE Feature Extraction Started")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Overlap size: {args.overlap_size}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Output directory: {args.save_dir}")
    logger.info("=" * 60)

    output_file = osp.join(args.save_dir, f"mae_feat_pjm_{split}.h5")

    processed_ids = set()
    if osp.exists(output_file):
        logger.info(f"Found existing file: {output_file}")
        with h5py.File(output_file, "r") as existing:
            processed_ids = set(existing.keys())
        logger.info(f"Resuming: {len(processed_ids)} videos already processed")
    else:
        logger.info(f"Will save to: {output_file}")

    generator, num = get_iterator_for_pjm(args, split=split, skip_ids=processed_ids)
    iterator = generator()

    if processed_ids:
        logger.info(f"Remaining: {num - len(processed_ids)} videos to process")
    else:
        logger.info(f"Processing {num} videos for split '{split}'")

    with h5py.File(name=output_file, mode="a") as f:
        if len(processed_ids) == 0:
            f.attrs["model"] = args.model_name
            f.attrs["overlap_size"] = args.overlap_size
            f.attrs["nth_layer"] = args.nth_layer
            f.attrs["dataset_name"] = "PJM"
            f.attrs["split"] = split
            f.attrs["num"] = num

        newly_processed = 0
        for mae_feat in iterator:
            feats, video_id, _ = mae_feat

            ds = f.create_dataset(
                video_id,
                data=feats,
                dtype="float32",
            )
            assert len(feats.shape) == 2, f"Expected 2D features, got {feats.shape}"
            ds.attrs["num_chunks"] = feats.shape[0]
            ds.attrs["features_dim"] = feats.shape[1]

            newly_processed += 1
            if newly_processed % save_in_every == 0:
                f.flush()
                logger.info(
                    f"Flushed at {len(processed_ids) + newly_processed}/{num} videos"
                )
        f.flush()

        logger.info(f"Newly processed {newly_processed} videos")

    elapsed_time = time.time() - start_time
    total_in_file = len(processed_ids) + newly_processed
    logger.info("=" * 60)
    logger.info("Feature Extraction Completed!")
    logger.info(f"  Total videos in output file: {total_in_file}/{num}")
    logger.info(f"  Newly processed in this run: {newly_processed}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total time: {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)")
    if newly_processed > 0:
        logger.info(f"  Average time per video: {elapsed_time / newly_processed:.2f}s")
    logger.info("=" * 60)


def main():
    save_hdf5(save_in_every=500)


if __name__ == "__main__":
    main()
