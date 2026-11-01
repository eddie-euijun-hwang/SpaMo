from lmdb import Environment
import msgpack
import numpy as np
from PIL.Image import Image
from PIL.ImageOps import pad as resize_and_pad
from io import BytesIO
from av import open as av_open


def get_video_frames(video_data: bytes) -> list[Image]:
    readable_bytes = BytesIO(video_data)
    container = av_open(readable_bytes)
    stream = container.streams.video[0]

    return [frame.to_image() for frame in container.decode(stream)]


class ImageConverter:
    def __init__(self, environment: Environment, output_size: tuple[int, int]):
        self.environment = environment
        self.output_size = output_size

    def _process_image(self, image: Image, crop_params: dict) -> Image:
        image = image.crop(
            (
                crop_params["x_start"],
                crop_params["y_start"],
                crop_params["x_end"],
                crop_params["y_end"],
            )
        )
        return resize_and_pad(image, self.output_size)

    def process_frames(self, frames: list[Image], id: str) -> np.array:
        with self.environment.begin(write=False) as transaction:
            byte_data = transaction.get(id.encode())
            if not byte_data:
                raise Exception(f"Crop params not found for {id}")
            crop_params = msgpack.unpackb(byte_data)

        return [self._process_image(frame, crop_params) for frame in frames]


if __name__ == "__main__":
    import lmdb
    import datasets

    dataset = datasets.load_dataset(
        "croco-corp/pjm-segments", split="train", streaming=True
    )
    rec = next(iter(dataset))
    frames = get_video_frames(rec["mp4"])
    env = lmdb.open("crop_params/crop_params.lmdb")
    converter = ImageConverter(env, (224, 224))
    processed = converter.process_frames(frames, rec["__key__"])
    print(processed[0].size)
    processed[0].show()
