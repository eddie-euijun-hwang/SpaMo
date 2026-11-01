import pandas as pd
import torch
import h5py
import random
import pytorch_lightning as pl


class PJMKorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        visual_features_path: str,
        motion_features_path: str,
        texts_path: str,
        device: str,
        max_frame_length: int = 512,
        keys: set | None = None,
        key_to_speaker: dict | None = None,
        keypoint_features_path: str | None = None,
        aux_features_path: str | None = None,
        multilang_path: str | None = None,
        target_lang: str = "English",
    ):
        super().__init__()

        self.device = device
        self.max_num_of_frames = max_frame_length
        self.visual_features_path = visual_features_path
        self.motion_features_path = motion_features_path
        self.texts_path = texts_path
        self.keypoint_features_path = keypoint_features_path
        self.aux_features_path = aux_features_path
        self.multilang_path = multilang_path
        self.target_lang = target_lang

        # Load only metadata (keys, texts, speaker ids) — features read lazily in __getitem__
        texts_h5 = h5py.File(texts_path, mode="r")
        visual_h5 = h5py.File(visual_features_path, mode="r")
        speaker_map = key_to_speaker or {}

        all_keys = sorted(visual_h5.keys())
        idx_to_key = [k for k in all_keys if k in keys] if keys else all_keys
        # TemporalConv conv_type=2 = [K5, P2, K5, P2] requires >= ~16 input frames.
        # Defensive filter — upstream PJM port crashed on short samples; fork pipeline
        # apparently coped historically but no harm filtering ~8% of MS samples.
        min_frames = 16
        idx_to_key = [k for k in idx_to_key if visual_h5[k].shape[0] >= min_frames]
        self.samples = [
            (key, texts_h5[key][()].decode(), speaker_map.get(key, "unknown"))
            for key in idx_to_key
        ]
        texts_h5.close()
        visual_h5.close()

        # H5 file handles — opened per worker in _open_files()
        self._visual_h5 = None
        self._motion_h5 = None
        self._keypoint_h5 = None
        self._aux_h5 = None
        self._multilang_h5 = None

    def _open_files(self):
        if self._visual_h5 is None:
            self._visual_h5 = h5py.File(self.visual_features_path, mode="r")
            self._motion_h5 = h5py.File(self.motion_features_path, mode="r")
            self._keypoint_h5 = (
                h5py.File(self.keypoint_features_path, mode="r")
                if self.keypoint_features_path
                else None
            )
            self._aux_h5 = (
                h5py.File(self.aux_features_path, mode="r")
                if self.aux_features_path
                else None
            )
            self._multilang_h5 = (
                h5py.File(self.multilang_path, mode="r")
                if self.multilang_path
                else None
            )

    def __getitem__(self, index: int) -> dict:
        self._open_files()
        key, text, speaker_id = self.samples[index]

        item_visual_features = torch.from_numpy(self._visual_h5[key][()])
        item_motion_features = torch.from_numpy(self._motion_h5[key][()])
        kp = (
            self._keypoint_h5[key][()]
            if (self._keypoint_h5 and key in self._keypoint_h5)
            else None
        )
        aux = self._aux_h5[key][()] if (self._aux_h5 and key in self._aux_h5) else None

        num_of_frames = item_visual_features.size(dim=0)
        if num_of_frames > self.max_num_of_frames:
            start_index = random.randint(0, num_of_frames - self.max_num_of_frames)
            item_visual_features = item_visual_features[
                start_index : start_index + self.max_num_of_frames
            ]

        out = {
            "id": key,
            "pixel_value": item_visual_features,
            "num_frames": item_visual_features.size(dim=0),
            "glor_value": item_motion_features,
            "keypoint_value": torch.from_numpy(kp) if kp is not None else None,
            "aux_value": torch.from_numpy(aux) if aux is not None else None,
            "text": text,
            "lang": self.target_lang,
            "speaker_id": speaker_id,
        }

        # Multilang in-context fields (used when t5_slt.use_in_context=True).
        # Layout: features/texts_multilang_pjm.h5 -> /{key}/{pl,en,fr,es}
        if self._multilang_h5 is not None and key in self._multilang_h5:
            grp = self._multilang_h5[key]
            out["en_text"] = grp["en"][()].decode() if "en" in grp else text
            out["fr_text"] = grp["fr"][()].decode() if "fr" in grp else text
            out["es_text"] = grp["es"][()].decode() if "es" in grp else text
            out["pl_text"] = grp["pl"][()].decode() if "pl" in grp else text

        return out

    def __len__(self) -> int:
        return len(self.samples)


def collate_data(batch):
    # Return the list of sample dicts as-is; get_inputs() handles batching internally
    return batch


class PJMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        visual_features_path: str,
        motion_features_path: str,
        texts_path: str,
        device: str,
        train_split_path: str,
        val_split_path: str,
        test_split_path: str | None = None,
        max_frame_length: int = 512,
        batch_size: int = 64,
        num_workers: int = 4,
        set_to_test: str = "val",
        keypoint_features_path: str | None = None,
        aux_features_path: str | None = None,
        multilang_path: str | None = None,
        target_lang: str = "English",
    ):
        super().__init__()
        self.visual_features_path = visual_features_path
        self.motion_features_path = motion_features_path
        self.texts_path = texts_path
        self.device = device
        self.train_split_path = train_split_path
        self.multilang_path = multilang_path
        self.target_lang = target_lang
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        self.max_frame_length = max_frame_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.set_to_test = set_to_test
        self.keypoint_features_path = keypoint_features_path
        self.aux_features_path = aux_features_path

    @staticmethod
    def _load_split(path: str):
        df = pd.read_csv(path)
        keys = set(df["key"].astype(str))
        key_to_speaker = dict(zip(df["key"].astype(str), df["speaker_id"].astype(str)))
        return keys, key_to_speaker

    def setup(self, stage=None):
        train_keys, train_k2s = self._load_split(self.train_split_path)
        val_keys, val_k2s = self._load_split(self.val_split_path)
        self.train_set = PJMKorpus(
            self.visual_features_path,
            self.motion_features_path,
            self.texts_path,
            device=self.device,
            max_frame_length=self.max_frame_length,
            keys=train_keys,
            key_to_speaker=train_k2s,
            keypoint_features_path=self.keypoint_features_path,
            aux_features_path=self.aux_features_path,
            multilang_path=self.multilang_path,
            target_lang=self.target_lang,
        )
        self.val_set = PJMKorpus(
            self.visual_features_path,
            self.motion_features_path,
            self.texts_path,
            device=self.device,
            max_frame_length=self.max_frame_length,
            keys=val_keys,
            key_to_speaker=val_k2s,
            keypoint_features_path=self.keypoint_features_path,
            aux_features_path=self.aux_features_path,
            multilang_path=self.multilang_path,
            target_lang=self.target_lang,
        )
        if self.test_split_path:
            test_keys, test_k2s = self._load_split(self.test_split_path)
            self.test_set = PJMKorpus(
                self.visual_features_path,
                self.motion_features_path,
                self.texts_path,
                device=self.device,
                max_frame_length=self.max_frame_length,
                keys=test_keys,
                key_to_speaker=test_k2s,
                keypoint_features_path=self.keypoint_features_path,
                aux_features_path=self.aux_features_path,
                multilang_path=self.multilang_path,
                target_lang=self.target_lang,
            )

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
        if self.set_to_test == "test" and hasattr(self, "test_set"):
            dataset = self.test_set
        elif self.set_to_test == "train":
            dataset = self.train_set
        else:
            dataset = self.val_set
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data,
        )


if __name__ == "__main__":
    dataset = PJMKorpus(
        "features/vit_feat_pjm.h5",
        "features/mae_feat_pjm.h5",
        "features/texts_eng.h5",
        device="cpu",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_data)
    batch = next(iter(loader))
    print(f"Batch size: {len(batch)}, keys: {list(batch[0].keys())}")
