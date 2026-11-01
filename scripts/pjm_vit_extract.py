import argparse
import tqdm
import torch
import numpy as np
from transformers import AutoImageProcessor, CLIPVisionModel
import datasets
import sys
import lmdb
from pathlib import Path
import h5py

sys.path.append("./")

from utils.s2wrapper import forward as multiscale_forward
from utils.pjm.preprocessing import get_video_frames, ImageConverter
from utils.log import get_error_logger

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

error_logger = get_error_logger("vit_extract_errors.log")


class ViTFeatureReader(object):
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        cache_dir=None,
        device="cuda:0",
        s2_mode="s2wrapping",
        scales=[1, 2],
        nth_layer=-1,
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer

        self.model = (
            CLIPVisionModel.from_pretrained(
                model_name, output_hidden_states=True, cache_dir=cache_dir
            )
            .to(device)
            .eval()
        )

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, do_center_crop=False, do_resize=False
        )
        print(self.image_processor.do_center_crop, self.image_processor.do_resize)

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        return outputs

    @torch.no_grad()
    def get_feats(self, video):
        inputs = (
            self.image_processor(list(video), return_tensors="pt")
            .to(self.device)
            .pixel_values
        )
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(
                self.forward_features, inputs, scales=self.scales, num_prefix_token=1
            )
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        help="Path to huggingface dataset",
        type=str,
        default="croco-corp/pjm-segments",
    )
    parser.add_argument(
        "--crop-params-path",
        help="Path to crop params lmdb file",
        type=str,
        default="crop_params/crop_params.lmdb",
    )
    parser.add_argument("--device", help="device to use", default="cuda:0")
    parser.add_argument("--s2_mode", default="")
    parser.add_argument(
        "--scales", nargs="+", type=int, help="List of scales", default=[]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nth_layer", type=int, default=-1)
    parser.add_argument("--cache_dir", help="cache dir for model", default=None)
    parser.add_argument(
        "--hdf5-save-interval",
        help="num of processed videos to flush to hdf5 file (Default: 500)",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--save_path", help="where to save the output", type=Path, required=True
    )
    parser.add_argument(
        "--model_name", help="ViT model name", default="openai/clip-vit-large-patch14"
    )

    return parser


class FeaturesExtractor:
    def __init__(
        self, converter: ImageConverter, reader: ViTFeatureReader, batch_size: int
    ):
        self.converter = converter
        self.reader = reader
        self.batch_size = batch_size

    def extract(self, record: dict[str, any]) -> np.ndarray:
        frames = get_video_frames(record["mp4"])
        processed = self.converter.process_frames(frames, record["__key__"])
        video_features = []
        for i in range(0, len(processed), self.batch_size):
            frame_batch = processed[i : i + self.batch_size]
            frame_features = self.reader.get_feats(frame_batch).cpu().numpy()
            video_features.append(frame_features)

        return np.concatenate(video_features, axis=0)


def main():
    parser = get_parser()
    args = parser.parse_args()

    batch_size = args.batch_size
    lmdb_env = lmdb.open(args.crop_params_path, readonly=True, lock=False)
    converter = ImageConverter(lmdb_env, (224, 224))

    reader = ViTFeatureReader(
        args.model_name,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
    )
    extractor = FeaturesExtractor(converter, reader, batch_size=batch_size)

    dataset = datasets.load_dataset(
        args.dataset_path, split="train", cache_dir=args.cache_dir
    )
    num = len(dataset)
    output_file: Path = args.save_path / "vit_feat_pjm.h5"
    processed_ids = set()
    if output_file.exists():
        with h5py.File(output_file, mode="r") as existing:
            processed_ids = set(existing.keys())
        dataset = dataset.filter(lambda record: record["__key__"] not in processed_ids)

    num_of_already_processed_videos = len(processed_ids)
    with h5py.File(name=output_file, mode="a") as hdf5_file:
        if num_of_already_processed_videos == 0:
            hdf5_file.attrs["model"] = args.model_name
            hdf5_file.attrs["nth_layer"] = args.nth_layer
            hdf5_file.attrs["dataset_name"] = "PJM"
            hdf5_file.attrs["split"] = "train"
            hdf5_file.attrs["num"] = num
            hdf5_file.attrs["s2_mode"] = args.s2_mode
            hdf5_file.attrs["scales"] = str(args.scales)

        pbar = tqdm.tqdm(total=num, desc="Processing PJM")
        if num_of_already_processed_videos != 0:
            pbar.update(num_of_already_processed_videos)

        errors_count = 0
        processed_count = 0
        for record in dataset:
            try:
                video_features = extractor.extract(record)

                assert len(video_features.shape) == 2, (
                    f"Expected 2D features, got {video_features.shape}"
                )

                ds = hdf5_file.create_dataset(
                    record["__key__"], data=video_features, dtype="float32"
                )
                ds.attrs["num_chunks"] = video_features.shape[0]
                ds.attrs["features_dim"] = video_features.shape[1]

                processed_count += 1
            except Exception:
                errors_count += 1
                failed_video_id = record["__key__"]
                error_logger.exception(f"extract features error for: {failed_video_id}")
            finally:
                if processed_count % args.hdf5_save_interval == 0:
                    hdf5_file.flush()
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "processed": processed_count + num_of_already_processed_videos,
                        "errors": errors_count,
                    }
                )
        hdf5_file.flush()
    lmdb_env.close()


if __name__ == "__main__":
    main()
