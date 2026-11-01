"""
Lightning DataModule for PHOENIX-2014T, producing batches in the same dict
format as dataset.pjm.PJMDataModule so spamo.t5_slt.FlanT5SLT can train on
Phoenix without code changes.

Streams expected on disk (paths configurable):
    spatial_root/{train,dev,test}/{file_id}.npy        # ViT [T, 2048]
    motion_root/{train,dev,test}/{file_id}.npy         # MAE [T, 1024]
    aux_features_path (HDF5)                           # hand-ViT [T, 2048]
    keypoint_features_path (HDF5)                      # MediaPipe [T, 258]

Annotations: preprocess/Phoenix14T/{train,dev,test}_info_ml.npy
"""

import random
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch


class Phoenix14TQuad(torch.utils.data.Dataset):
    def __init__(
        self,
        anno_path: str,
        spatial_root: str,
        motion_root: str,
        keypoint_features_path: str | None = None,
        aux_features_path: str | None = None,
        max_frame_length: int = 512,
        text_field: str = "en_text",
    ):
        super().__init__()
        anno = np.load(anno_path, allow_pickle=True).item()
        self.spatial_root = Path(spatial_root)
        self.motion_root = Path(motion_root)
        self.max_frame_length = max_frame_length
        self.text_field = text_field
        self.keypoint_features_path = keypoint_features_path
        self.aux_features_path = aux_features_path

        # When H5 streams are configured AND the file already exists, restrict
        # entries to keys present in those H5s — prevents partial-batch issues
        # when extraction failed on some videos (t5_slt.py:442-450 silently drops
        # samples where keypoint/aux are None, breaking batch alignment).
        required_h5_keys = None

        def _intersect(curr, path):
            if not path or not Path(path).exists():
                return curr
            with h5py.File(path, "r") as f:
                ks = set(f.keys())
            return ks if curr is None else curr & ks

        required_h5_keys = _intersect(required_h5_keys, keypoint_features_path)
        required_h5_keys = _intersect(required_h5_keys, aux_features_path)

        self.entries = []
        for i in sorted(k for k in anno.keys() if isinstance(k, int)):
            d = anno[i]
            fid = d["fileid"]
            if not (self.spatial_root / f"{fid}.npy").exists():
                continue
            if not (self.motion_root / f"{fid}.npy").exists():
                continue
            if required_h5_keys is not None and fid not in required_h5_keys:
                continue
            self.entries.append(d)

        # H5 handles — opened lazily per worker
        self._kp_h5 = None
        self._aux_h5 = None

    def _open_h5(self):
        if self.keypoint_features_path and self._kp_h5 is None:
            self._kp_h5 = h5py.File(self.keypoint_features_path, "r")
        if self.aux_features_path and self._aux_h5 is None:
            self._aux_h5 = h5py.File(self.aux_features_path, "r")

    def __getitem__(self, idx: int) -> dict:
        self._open_h5()
        d = self.entries[idx]
        fid = d["fileid"]

        vit = torch.from_numpy(np.load(self.spatial_root / f"{fid}.npy"))
        mae = torch.from_numpy(np.load(self.motion_root / f"{fid}.npy"))

        # Mirror PJMKorpus: random-crop spatial if longer than max_frame_length.
        n = vit.size(0)
        if n > self.max_frame_length:
            start = random.randint(0, n - self.max_frame_length)
            vit = vit[start : start + self.max_frame_length]

        kp = None
        if self._kp_h5 is not None and fid in self._kp_h5:
            kp = torch.from_numpy(self._kp_h5[fid][()])

        aux = None
        if self._aux_h5 is not None and fid in self._aux_h5:
            aux = torch.from_numpy(self._aux_h5[fid][()])

        text = (d.get(self.text_field) or d["text"]).strip()
        if not text.endswith("."):
            text += "."

        # use_in_context=True path in t5_slt.py:413-419 reads en_text/fr_text/es_text
        # off the sample. Pass them through verbatim if present in annotations.
        return {
            "id": fid,
            "pixel_value": vit,
            "num_frames": vit.size(0),
            "glor_value": mae,
            "keypoint_value": kp,
            "aux_value": aux,
            "text": text,
            "en_text": d.get("en_text", text),
            "fr_text": d.get("fr_text", text),
            "es_text": d.get("es_text", text),
            "lang": "German",
            "speaker_id": d.get("signer", "unknown"),
        }

    def __len__(self) -> int:
        return len(self.entries)


def collate_data(batch):
    return batch


class Phoenix14TDataModule(pl.LightningDataModule):
    def __init__(
        self,
        anno_root: str,
        spatial_root: str,
        motion_root: str,
        device: str = "cuda",
        max_frame_length: int = 512,
        batch_size: int = 4,
        num_workers: int = 4,
        set_to_test: str = "test",
        keypoint_features_path: str | None = None,
        aux_features_path: str | None = None,
        text_field: str = "en_text",
    ):
        super().__init__()
        self.anno_root = Path(anno_root)
        self.spatial_root = Path(spatial_root)
        self.motion_root = Path(motion_root)
        self.device = device
        self.max_frame_length = max_frame_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.set_to_test = set_to_test
        self.keypoint_features_path = keypoint_features_path
        self.aux_features_path = aux_features_path
        self.text_field = text_field

    def _make(self, mode: str) -> Phoenix14TQuad:
        return Phoenix14TQuad(
            anno_path=str(self.anno_root / f"{mode}_info_ml.npy"),
            spatial_root=str(self.spatial_root / mode),
            motion_root=str(self.motion_root / mode),
            keypoint_features_path=self.keypoint_features_path,
            aux_features_path=self.aux_features_path,
            max_frame_length=self.max_frame_length,
            text_field=self.text_field,
        )

    def setup(self, stage=None):
        self.train_set = self._make("train")
        self.val_set = self._make("dev")
        self.test_set = self._make("test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_data,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data,
        )

    def test_dataloader(self):
        ds = {"train": self.train_set, "val": self.val_set, "test": self.test_set}.get(
            self.set_to_test, self.val_set
        )
        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data,
        )


if __name__ == "__main__":
    dm = Phoenix14TDataModule(
        anno_root="preprocess/Phoenix14T",
        spatial_root="features/p14t/spatial",
        motion_root="features/p14t/motion",
        batch_size=2,
        num_workers=0,
    )
    dm.setup()
    print(
        f"train: {len(dm.train_set)} | dev: {len(dm.val_set)} | test: {len(dm.test_set)}"
    )
    sample = dm.train_set[0]
    print(
        {
            k: (tuple(v.shape) if torch.is_tensor(v) else v)
            for k, v in sample.items()
            if v is not None
        }
    )
